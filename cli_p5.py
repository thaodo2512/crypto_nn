from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from folds import make_purged_folds


app = typer.Typer(help="P5 – Small NN training (GRU) for 5m BTCUSDT")


def _read_parquet(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _select_features(df: pd.DataFrame) -> List[str]:
    exclude = {"ts", "symbol", "label"}
    cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in {"f", "i"}]
    return cols


def build_windows(features: pd.DataFrame, labels: pd.DataFrame, W: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = features.sort_values(["symbol", "ts"]).reset_index(drop=True)
    labs = labels.sort_values(["symbol", "ts"]).reset_index(drop=True)
    cols = _select_features(df)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[dict] = []
    mapping = {"WAIT": 0, "LONG": 1, "SHORT": 2}
    for sym, g in df.groupby("symbol", sort=False):
        g = g.set_index("ts").sort_index()
        lab_sym = labs[labs["symbol"] == sym]
        for _, row in lab_sym.iterrows():
            t = row["ts"]
            win = g.loc[(g.index > t - pd.Timedelta(minutes=5 * W)) & (g.index <= t), cols]
            if len(win) != W:
                continue
            X_list.append(win.values[np.newaxis, :, :])
            y_list.append(mapping.get(str(row["label"]).upper(), 0))
            meta_rows.append({"ts": t, "symbol": sym})
    if not X_list:
        return np.empty((0, W, 0)), np.array([], dtype=int), pd.DataFrame(columns=["ts", "symbol"])  # type: ignore
    X = np.vstack(X_list)  # [N, W, F]
    y = np.array(y_list, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2, n_layers: int = 1, n_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, F]
        out, h = self.gru(x)
        last = out[:, -1, :]
        z = self.drop(last)
        logits = self.fc(z)
        return logits


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    model: str
    window: int
    embargo: str
    out_dir: str
    seed: int


def _time_decay_weights(ts: pd.Series, lam_per_day: float = 0.98) -> np.ndarray:
    tmax = ts.max()
    days = (tmax - ts).dt.total_seconds().to_numpy() / (3600 * 24)
    w = np.power(lam_per_day, days)
    return w.astype(np.float32)


def _train_fold(model_dir: Path, X: np.ndarray, y: np.ndarray, ts: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray, class_weights: torch.Tensor, seed: int) -> Dict[str, float]:
    device = torch.device("cpu")
    input_dim = X.shape[2]
    net = GRUClassifier(input_dim=input_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="none")

    # Standardize per feature using train set
    mu = X[train_idx].mean(axis=(0, 1))
    std = X[train_idx].std(axis=(0, 1))
    std[std == 0] = 1.0
    Xn = (X - mu) / std

    def run_epoch(idx: np.ndarray) -> Tuple[float, float]:
        net.train()
        total_loss = 0.0
        n = 0
        # Simple batching
        bs = 64
        for start in range(0, len(idx), bs):
            sl = idx[start : start + bs]
            xb = torch.tensor(Xn[sl], dtype=torch.float32, device=device)
            yb = torch.tensor(y[sl], dtype=torch.long, device=device)
            logits = net(xb)
            # per-sample loss with class weights and time-decay
            l = ce(logits, yb)
            cw = class_weights[yb]
            td = torch.tensor(_time_decay_weights(ts.iloc[sl]), dtype=torch.float32, device=device)
            loss = (l * cw * td).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Guard against NaN/inf gradients
            grad_ok = True
            for p in net.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        grad_ok = False
                        break
            if grad_ok:
                opt.step()
            total_loss += float(loss.item()) * len(sl)
            n += len(sl)
        return total_loss / max(n, 1), 0.0

    @torch.no_grad()
    def eval_epoch(idx: np.ndarray) -> Tuple[float, Dict[str, float]]:
        net.eval()
        bs = 256
        total = 0.0
        n = 0
        preds = []
        ys = []
        for start in range(0, len(idx), bs):
            sl = idx[start : start + bs]
            xb = torch.tensor(Xn[sl], dtype=torch.float32, device=device)
            yb = torch.tensor(y[sl], dtype=torch.long, device=device)
            logits = net(xb)
            l = ce(logits, yb)
            cw = class_weights[yb]
            td = torch.tensor(_time_decay_weights(ts.iloc[sl]), dtype=torch.float32, device=device)
            loss = (l * cw * td).mean()
            total += float(loss.item()) * len(sl)
            n += len(sl)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(yb.cpu().numpy())
        preds = np.concatenate(preds) if preds else np.array([])
        ys = np.concatenate(ys) if ys else np.array([])
        # Precision/recall placeholders
        def pr(label: int) -> Tuple[float, float]:
            tp = ((preds == label) & (ys == label)).sum()
            fp = ((preds == label) & (ys != label)).sum()
            fn = ((preds != label) & (ys == label)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            return float(prec), float(rec)

        p_long, r_long = pr(1)
        p_short, r_short = pr(2)
        metrics = {
            "val_loss": total / max(n, 1),
            "precision_long": p_long,
            "recall_long": r_long,
            "precision_short": p_short,
            "recall_short": r_short,
            "EV": 0.0,
            "ECE": 0.0,
        }
        return metrics["val_loss"], metrics

    best = math.inf
    best_metrics: Dict[str, float] = {}
    epochs = 3
    for ep in range(1, epochs + 1):
        tr_loss, _ = run_epoch(train_idx)
        val_loss, metrics = eval_epoch(val_idx)
        metrics["train_loss"] = tr_loss
        if val_loss < best and np.isfinite(val_loss):
            best = val_loss
            best_metrics = metrics
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict()}, model_dir / "best.pt")
    return best_metrics


@app.command("train")
def train(
    model: str = typer.Option("gru", "--model"),
    window: int = typer.Option(144, "--window"),
    cv: str = typer.Option("walkforward", "--cv"),
    embargo: str = typer.Option("1D", "--embargo"),
    features: str = typer.Option(..., "--features"),
    labels: str = typer.Option(..., "--labels"),
    out: str = typer.Option("models/gru_5m", "--out"),
    seed: int = typer.Option(42, "--seed"),
    folds_n: int = typer.Option(5, "--folds"),
) -> None:
    logger = logging.getLogger("p5")
    Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler("logs/p5_train.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    _set_seed(seed)
    feat = _read_parquet(features)
    lab = _read_parquet(labels)
    X, y, meta = build_windows(feat, lab, W=window)
    if X.shape[0] == 0:
        raise typer.BadParameter("No windows built; check inputs or window length")

    # Folds on label timestamps
    ts_sorted = meta.sort_values("ts")["ts"]
    folds = make_purged_folds(ts_sorted, n_folds=folds_n, embargo=embargo)
    # Prepare class weights from overall train distribution (approx)
    class_counts = np.bincount(y, minlength=3)
    weights = class_counts.sum() / np.maximum(1, class_counts)
    class_weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)

    metrics_cv: Dict[str, Dict[str, float]] = {}
    for f in folds:
        fid = f["fold_id"]
        tr_idx = f["train_idx"]
        vl_idx = f["val_idx"]
        # Map fold indices to meta ordering
        # Our folds were built on sorted ts; meta is not guaranteed sorted identically
        meta_sorted = meta.sort_values("ts").reset_index(drop=True)
        m_tr_ts = meta_sorted.iloc[tr_idx]["ts"]
        m_vl_ts = meta_sorted.iloc[vl_idx]["ts"]
        sel_tr = meta["ts"].isin(set(m_tr_ts))
        sel_vl = meta["ts"].isin(set(m_vl_ts))
        tr_ids = np.where(sel_tr.to_numpy())[0]
        vl_ids = np.where(sel_vl.to_numpy())[0]
        fold_dir = Path(out) / str(fid)
        m = _train_fold(fold_dir, X, y, meta["ts"], tr_ids, vl_ids, class_weights, seed)
        metrics_cv[str(fid)] = m
        logger.info(f"fold={fid} train_loss={m.get('train_loss', 0):.4f} val_loss={m.get('val_loss', 0):.4f}")

    # Save metrics
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/p5_cv_metrics.json", "w") as f:
        json.dump(metrics_cv, f, indent=2)
    typer.echo("P5 training completed; metrics saved.")


if __name__ == "__main__":
    app()

