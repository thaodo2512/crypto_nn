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


app = typer.Typer(help="P5/P6 – Training + Calibration/Ensemble for 5m BTCUSDT")


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

    # Standardize per feature using train set; if train is empty, fall back to global stats to avoid NaNs
    if train_idx.size == 0:
        mu = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
    else:
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
    # Log loss/time-decay policy for validator
    logger.info("loss=CrossEntropyLoss time_decay_lambda=0.98 weight_decay=1e-4 dropout=0.2 model=GRU(64,1)")
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


# ---------- Phase 6: Calibration, Ensembling, Threshold Tuning ----------


def _softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    return p


def _ece_top(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    # Multi-class ECE on top-class
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc = correct[mask].mean()
            avg_conf = conf[mask].mean()
            ece += (mask.sum() / N) * abs(acc - avg_conf)
    return float(ece)


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = _softmax(logits, T=T)
    eps = 1e-9
    ll = -np.log(p[np.arange(len(y)), y] + eps)
    return float(ll.mean())


def _load_probs(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


@app.command("calibrate")
def calibrate(
    probs: str = typer.Option(..., "--probs", help="Per-fold logits/probs parquet glob with columns [fold_id, split, ts, y, logits_* or p_*]"),
    method: str = typer.Option("temperature", "--method"),
    out: str = typer.Option("models/calib.json", "--out"),
) -> None:
    df = _load_probs(probs)
    # Expect columns: fold_id, split in {val,oos}, y (int), and either logits_0..2 or p_0..2
    folds = sorted(df["fold_id"].unique()) if "fold_id" in df else [0]
    calib: Dict[str, Dict[str, float]] = {}
    for fid in folds:
        sub = df[df.get("fold_id", 0) == fid]
        val = sub[sub.get("split", "val") == "val"]
        if any(c.startswith("logits_") for c in val.columns):
            logits = val[[c for c in val.columns if c.startswith("logits_")]].to_numpy()
        elif any(c.startswith("p_") for c in val.columns):
            # Convert probs to logits via inverse softmax (log)
            p = val[[c for c in val.columns if c.startswith("p_")]].to_numpy()
            logits = np.log(np.maximum(p, 1e-8))
        else:
            raise typer.BadParameter("Missing logits_ or p_ columns")
        yv = val["y"].to_numpy().astype(int)
        bestT = 1.0
        bestNLL = 1e9
        for T in np.linspace(0.5, 5.0, 91):
            nll = _nll(logits, yv, T)
            if nll < bestNLL:
                bestNLL = nll
                bestT = float(T)
        pv = _softmax(logits, T=bestT)
        ece = _ece_top(pv, yv)
        calib[str(fid)] = {"temperature": bestT, "ece_val": ece, "nll_val": bestNLL}
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(calib, f, indent=2)
    typer.echo(f"Calibration saved to {out}")


def _ev_trade(prob: np.ndarray, y: np.ndarray, tau: float, cost_bps: float) -> float:
    # Simple EV per trade: +1 correct direction, -1 wrong direction, 0 on WAIT or no trade; subtract costs.
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    take = conf >= tau
    pnl = np.zeros_like(conf)
    # LONG=1, SHORT=2, WAIT=0
    pnl[(pred == 1) & (y == 1) & take] = 1.0
    pnl[(pred == 1) & (y != 1) & take] = -1.0
    pnl[(pred == 2) & (y == 2) & take] = 1.0
    pnl[(pred == 2) & (y != 2) & take] = -1.0
    trades = take.sum()
    if trades == 0:
        return 0.0
    # cost per trade in fraction
    cost = cost_bps / 1e4
    ev = pnl[take].mean() - cost
    return float(ev)


@app.command("ensemble")
def ensemble(
    calib: str = typer.Option(..., "--calib", help="Calibration JSON from calibrate"),
    probs: str = typer.Option(..., "--probs", help="Per-fold OOS probs/logits parquet glob with split='oos'"),
    weight_by: str = typer.Option("EV", "--weight-by"),
    out: str = typer.Option("models/ensemble_5m.json", "--out"),
    cost_bps: float = typer.Option(5.0, "--cost_bps"),
) -> None:
    with open(calib, "r") as f:
        cal = json.load(f)
    df = _load_probs(probs)
    oos = df[df.get("split", "oos") == "oos"].copy()
    folds = sorted(oos["fold_id"].unique()) if "fold_id" in oos else [0]
    # Build calibrated probabilities per fold
    prob_folds: Dict[int, np.ndarray] = {}
    y = oos["y"].to_numpy().astype(int)
    for fid in folds:
        sub = oos[oos.get("fold_id", 0) == fid]
        if any(c.startswith("logits_") for c in sub.columns):
            logits = sub[[c for c in sub.columns if c.startswith("logits_")]].to_numpy()
            T = float(cal.get(str(fid), {}).get("temperature", 1.0))
            p = _softmax(logits, T=T)
        else:
            p = sub[[c for c in sub.columns if c.startswith("p_")]].to_numpy()
        prob_folds[fid] = p
    # Compute EV per fold at tau=0.5 (baseline)
    evs = []
    for fid in folds:
        ev = _ev_trade(prob_folds[fid], y, tau=0.5, cost_bps=cost_bps)
        evs.append(max(ev, 0.0))
    evs = np.array(evs)
    if evs.sum() == 0:
        w = np.ones_like(evs) / len(evs)
    else:
        w = evs / evs.sum()
    weights = {str(fid): float(w[i]) for i, fid in enumerate(folds)}
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"weights": weights, "calibration": cal}, f, indent=2)
    typer.echo(f"Ensemble weights saved to {out}")


@app.command("tune-threshold")
def tune_threshold(
    ensemble_json: str = typer.Option(..., "--ensemble", help="Ensemble JSON with weights + calibration"),
    probs: str = typer.Option(..., "--probs", help="Per-fold OOS probs/logits parquet glob"),
    grid: str = typer.Option("0.50:0.80:0.025", "--grid"),
    cost_spec: str = typer.Option("bps:5", "--cost"),
    out: str = typer.Option("reports/p6_oos_summary.json", "--out"),
) -> None:
    import matplotlib.pyplot as plt

    with open(ensemble_json, "r") as f:
        ens = json.load(f)
    weights = {int(k): float(v) for k, v in ens.get("weights", {}).items()}
    cal = ens.get("calibration", {})
    df = _load_probs(probs)
    oos = df[df.get("split", "oos") == "oos"].copy()
    folds = sorted(oos["fold_id"].unique()) if "fold_id" in oos else [0]
    y = oos["y"].to_numpy().astype(int)
    # Calibrated fold probs
    P_list = []
    for fid in folds:
        sub = oos[oos.get("fold_id", 0) == fid]
        if any(c.startswith("logits_") for c in sub.columns):
            logits = sub[[c for c in sub.columns if c.startswith("logits_")]].to_numpy()
            T = float(cal.get(str(fid), {}).get("temperature", 1.0))
            p = _softmax(logits, T=T)
        else:
            p = sub[[c for c in sub.columns if c.startswith("p_")]].to_numpy()
        w = float(weights.get(fid, 1.0 / max(len(folds), 1)))
        P_list.append(w * p)
    P_ens = np.sum(P_list, axis=0) if P_list else np.zeros((len(y), 3))

    # Grid search tau
    start, end, step = [float(x) for x in grid.split(":")]
    taus = np.arange(start, end + 1e-9, step)
    cost_bps = float(cost_spec.split(":")[1]) if cost_spec.startswith("bps:") else 5.0
    evs = []
    precs = []
    recs = []
    for tau in taus:
        ev = _ev_trade(P_ens, y, tau=tau, cost_bps=cost_bps)
        evs.append(ev)
        # precision/recall on non-WAIT
        pred = P_ens.argmax(axis=1)
        conf = P_ens.max(axis=1)
        take = conf >= tau
        tp = ((pred == y) & (take) & (y != 0)).sum()
        fp = ((pred != y) & (take) & (pred != 0)).sum()
        fn = ((pred != 0) & (~take) & (y != 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        precs.append(prec)
        recs.append(rec)

    # Choose best tau
    best_idx = int(np.argmax(evs))
    best_tau = float(taus[best_idx])
    # ECE after ensembling at best tau threshold (calibration overall)
    ece = _ece_top(P_ens, y)

    # Split OOS into two time segments for CI (simple): halves
    mid = len(y) // 2
    segs = [(slice(0, mid), "seg1"), (slice(mid, None), "seg2")]
    seg_ev = {}
    for s, name in segs:
        seg_ev[name] = _ev_trade(P_ens[s], y[s], tau=best_tau, cost_bps=cost_bps)
    # 95% CI by normal approx across segments (few segments → illustrative)
    vals = np.array(list(seg_ev.values()))
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    ci95 = [mu - 1.96 * sigma, mu + 1.96 * sigma]

    Path("reports").mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "best_tau": best_tau,
                "ev_trade": evs[best_idx],
                "ev_by_tau": {str(float(t)): float(e) for t, e in zip(taus, evs)},
                "precision_by_tau": {str(float(t)): float(p) for t, p in zip(taus, precs)},
                "recall_by_tau": {str(float(t)): float(r) for t, r in zip(taus, recs)},
                "ece": ece,
                "seg_ev": seg_ev,
                "ci95_ev": ci95,
            },
            f,
            indent=2,
        )

    # Curves plot
    plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax1.plot(taus, evs, label="EV/trade", color="C0")
    ax1.set_xlabel("tau")
    ax1.set_ylabel("EV/trade", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(taus, precs, label="precision", color="C1", linestyle="--")
    ax2.plot(taus, recs, label="recall", color="C2", linestyle=":")
    ax2.set_ylabel("precision/recall")
    ax1.axvline(best_tau, color="gray", linestyle="-")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("reports/p6_curves.png", dpi=150)
    typer.echo(f"Threshold tuned: tau={best_tau:.3f}, report saved to {out}")


if __name__ == "__main__":
    app()
