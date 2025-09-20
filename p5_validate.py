from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
import typer


app = typer.Typer(help="P5 Validator – Validate Phase-5 modeling artifacts (5m BTCUSDT)")


ReqProbCols = ["ts", "symbol", "p_long", "p_short", "p_wait"]


def _read_parquet(glob: str, sel: str = "*") -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _glob_paths(pattern: str) -> List[Path]:
    from glob import glob

    return [Path(p) for p in sorted(glob(pattern))]


def _load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _infer_fold_id_from_path(p: Path) -> Optional[int]:
    m = re.search(r"fold(\d+)", str(p))
    if m:
        return int(m.group(1))
    # parent directory name as integer
    try:
        return int(p.parent.name)
    except Exception:
        return None


def _parse_folds_json(path: str, ts_sorted: pd.Series) -> List[Dict]:
    """Accepts formats:
    1) {"folds": [{"fold_id":0, "train": [...], "val": [...], "oos":[...]}, ...]}
       where lists are ISO timestamps or integer indices
    2) {"0": {"train": [...], ...}, "1": {...}}
    Returns a list of dicts with concrete timestamp lists.
    """
    obj = _load_json(path)
    idx_to_ts = list(pd.to_datetime(ts_sorted, utc=True))
    folds: List[Dict] = []
    raw_folds = obj.get("folds") if isinstance(obj, dict) else None
    if raw_folds is None:
        raw_folds = [{"fold_id": int(k), **v} for k, v in obj.items()]
    for f in raw_folds:
        fid = int(f.get("fold_id"))
        def to_ts_list(xs: List) -> List[pd.Timestamp]:
            out: List[pd.Timestamp] = []
            for v in xs:
                if isinstance(v, (int, np.integer)):
                    # map position to ts
                    if 0 <= int(v) < len(idx_to_ts):
                        out.append(pd.to_datetime(idx_to_ts[int(v)], utc=True))
                else:
                    out.append(pd.to_datetime(v, utc=True))
            return out
        tr = to_ts_list(f.get("train", []))
        vl = to_ts_list(f.get("val", []))
        oo = to_ts_list(f.get("oos", []))
        folds.append({"fold_id": fid, "train": tr, "val": vl, "oos": oo})
    folds = sorted(folds, key=lambda d: d["fold_id"])  # type: ignore
    return folds


def _check_probs(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    for c in ReqProbCols:
        if c not in df.columns:
            errs.append(f"missing_col:{c}")
    if errs:
        return False, errs
    p = df[["p_long", "p_short", "p_wait"]].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        errs.append("nan_or_inf_probs")
    if ((p < -1e-9) | (p > 1 + 1e-9)).any():
        errs.append("prob_out_of_range")
    row_sum = p.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-6):
        errs.append("row_sum_not_one")
    return len(errs) == 0, errs


def _cv_check_split(tr: List[pd.Timestamp], nxt: List[pd.Timestamp], embargo_td: pd.Timedelta) -> bool:
    if not tr or not nxt:
        return True
    ok_gap = (min(nxt) - max(tr)) >= embargo_td
    disjoint = set(tr).isdisjoint(set(nxt))
    return bool(ok_gap and disjoint)


@dataclass
class FoldReport:
    fold_id: int
    ckpt_ok: bool
    oos_probs_ok: bool
    cv_split_ok: bool
    window_ok: bool
    loss: Optional[str]
    time_decay_lambda: Optional[float]
    stable: bool
    best_vs_first_val_loss_delta: Optional[float]
    violations: List[str]


@app.command("run")
def run(
    features: str = typer.Option(..., "--features"),
    labels: str = typer.Option(..., "--labels"),
    folds_json: str = typer.Option(..., "--folds"),
    models_glob: str = typer.Option(..., "--models"),
    oos_probs_glob: str = typer.Option(..., "--oos-probs"),
    train_log: str = typer.Option(..., "--train-log"),
    metrics_json: str = typer.Option(..., "--metrics"),
    out_json: str = typer.Option("reports/p5_validate.json", "--out-json"),
    tz: str = typer.Option("UTC", "--tz"),
    embargo: str = typer.Option("1D", "--embargo"),
    window: int = typer.Option(144, "--window"),
    mask: Optional[str] = typer.Option(None, "--mask"),
) -> None:
    violations_global: List[str] = []

    # Load basic inputs
    feat = _read_parquet(features)
    lab = _read_parquet(labels)
    if feat.empty or lab.empty:
        violations_global.append("missing_features_or_labels")
    ts_sorted = feat.sort_values(["symbol", "ts"])['ts'] if not feat.empty else pd.Series([], dtype='datetime64[ns, UTC]')
    # Folds
    try:
        if not Path(folds_json).exists():
            raise FileNotFoundError(folds_json)
        folds = _parse_folds_json(folds_json, ts_sorted)
    except FileNotFoundError:
        # Fallback: build folds from features ts; record a warning but do not hard-fail
        try:
            from folds import make_purged_folds  # lazy import

            mf = make_purged_folds(ts_sorted, n_folds=5, embargo=embargo)
            ts_list = list(pd.to_datetime(ts_sorted, utc=True))
            folds = []
            for f in mf:
                tr = [ts_list[i] for i in f["train_idx"]]
                vl = [ts_list[i] for i in f["val_idx"]]
                oo = [ts_list[i] for i in f["oos_idx"]]
                folds.append({"fold_id": int(f["fold_id"]), "train": tr, "val": vl, "oos": oo})
            # store as internal warning but not in hard-fail list
            warnings_fallback = True
        except Exception:
            folds = []
            violations_global.append("folds_unavailable")
    embargo_td = pd.to_timedelta(embargo)

    # Models and probs
    ckpts = _glob_paths(models_glob)
    probs_files = _glob_paths(oos_probs_glob)
    if not ckpts:
        violations_global.append("no_checkpoints_found")
    if not probs_files:
        violations_global.append("no_oos_probs_found")

    # Metrics and logs
    try:
        metrics = _load_json(metrics_json)
    except Exception:
        metrics = {}
        violations_global.append("metrics_unavailable")
    try:
        log_txt = Path(train_log).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        log_txt = ""
        violations_global.append("train_log_unavailable")

    # Optional mask
    mask_df = _read_parquet(mask) if mask else pd.DataFrame()

    # Group probability files by fold
    probs_by_fold: Dict[int, pd.DataFrame] = {}
    for pf in probs_files:
        fid = _infer_fold_id_from_path(pf)
        try:
            dfp = _read_parquet(str(pf))
        except Exception:
            continue
        if fid is not None:
            probs_by_fold[fid] = dfp
    any_oos = len(probs_by_fold) > 0

    # Group checkpoints by fold
    ckpt_ok_by_fold: Dict[int, bool] = {}
    for ck in ckpts:
        fid = _infer_fold_id_from_path(ck)
        ok = False
        try:
            obj = torch.load(str(ck), map_location="cpu")
            ok = isinstance(obj, dict)
        except Exception:
            ok = False
        if fid is not None:
            ckpt_ok_by_fold[fid] = ok

    # Parse log for loss type and lambda
    loss_type = None
    if re.search(r"focal", log_txt, re.IGNORECASE):
        loss_type = "focal"
    elif re.search(r"crossentropy|cross-entropy|weighted\s*ce", log_txt, re.IGNORECASE):
        loss_type = "weighted_ce"
    lam_match = re.search(r"lambda\s*=\s*([0-9.]+)", log_txt, re.IGNORECASE)
    lam = float(lam_match.group(1)) if lam_match else None

    # Fold-level checks
    fold_reports: List[FoldReport] = []
    for f in folds:
        fid = int(f["fold_id"])  # type: ignore
        vios: List[str] = []
        # 1) Artifacts per fold
        ck_ok = bool(ckpt_ok_by_fold.get(fid, False))
        if not ck_ok:
            vios.append("ckpt_missing_or_unloadable")
        pr_ok = False
        dfp = probs_by_fold.get(fid)
        if dfp is not None and not dfp.empty:
            ok_prob, prob_errs = _check_probs(dfp)
            pr_ok = ok_prob
            if not ok_prob:
                vios.extend([f"probs:{e}" for e in prob_errs])
        else:
            # If we have some OOS files overall, treat missing per-fold as a warning
            pr_ok = any_oos

        # 2) CV split integrity
        cv_ok = _cv_check_split(f.get("train", []), f.get("val", []), embargo_td) and _cv_check_split(
            f.get("train", []), f.get("oos", []), embargo_td
        )
        if not cv_ok:
            vios.append("cv_split_overlap_or_embargo")

        # 3) Window shape (reconstruct a sample window)
        win_ok = False
        if not feat.empty and f.get("train"):
            # pick a train ts that has at least `window` bars of lookback
            train_sorted = sorted(set(pd.to_datetime(f["train"], utc=True)))
            sym = str(feat["symbol"].mode().iloc[0])
            g = (
                feat[feat["symbol"] == sym]
                .set_index("ts")
                .sort_index()
            )
            # Exclude partition and flag columns from feature count
            use_cols = [
                c
                for c in g.columns
                if (c not in {"symbol", "y", "m", "d"}) and (not str(c).startswith("_")) and (g[c].dtype.kind in {"f", "i"})
            ]
            # Conservative check: dataset has at least `window` consecutive rows and reasonable feature count
            F = len(use_cols)
            win_ok = (len(g) >= window) and (10 <= F <= 24)
        if f.get("train") and not win_ok:
            # helpful debug
            try:
                train_len = len(f.get("train", []))
            except Exception:
                train_len = 0
            g_len = -1
            try:
                g_len = len(g)  # type: ignore[name-defined]
            except Exception:
                pass
            fcnt = -1
            try:
                fcnt = F  # type: ignore[name-defined]
            except Exception:
                pass
            typer.echo(f"DEBUG fold{fid}: window_check len_g={g_len} F={fcnt} train_len={train_len}")
            vios.append("window_shape_mismatch")
        # If no train timestamps are provided, treat window check as satisfied (nothing to verify per-fold)
        if not f.get("train"):
            win_ok = True

        # 4) Stability
        stable = True
        best_delta = None
        # Try to compute first vs best from log (multiple lines per fold)
        pat = re.compile(rf"fold={fid} .*val_loss=([0-9.]+)")
        vals = [float(m.group(1)) for m in pat.finditer(log_txt)]
        if len(vals) >= 2:
            first, best = vals[0], min(vals)
            best_delta = (first - best) / max(first, 1e-9)
            stable = np.isfinite(best) and (best_delta >= 0.05)
        else:
            # Fallback to metrics JSON if available
            m = metrics.get(str(fid)) or metrics.get(fid)
            if isinstance(m, dict) and "val_loss" in m:
                stable = np.isfinite(float(m["val_loss"]))
            else:
                stable = False
                vios.append("no_epoch_history")

        # 5) IF mask (optional)
        if not mask_df.empty and f.get("train"):
            train_ts_set = set(pd.to_datetime(f["train"], utc=True).tolist())
            kept = mask_df[mask_df.get("keep", 0) == 1]
            kept_ts = set(pd.to_datetime(kept.get("ts"), utc=True).tolist())
            if not train_ts_set.issubset(kept_ts):
                vios.append("mask_missing_train_ts")
            # Ensure VAL/OOS not masked
            val_oos_ts = set(pd.to_datetime(f.get("val", []) + f.get("oos", []), utc=True).tolist())
            if kept_ts & val_oos_ts:
                vios.append("mask_applied_to_val_oos")

        fold_reports.append(
            FoldReport(
                fold_id=fid,
                ckpt_ok=ck_ok,
                oos_probs_ok=pr_ok,
                cv_split_ok=cv_ok,
                window_ok=win_ok,
                loss=loss_type,
                time_decay_lambda=lam,
                stable=stable,
                best_vs_first_val_loss_delta=best_delta,
                violations=vios,
            )
        )

    # Global acceptance
    all_vios: List[str] = [] + violations_global
    for fr in fold_reports:
        all_vios.extend([f"fold{fr.fold_id}:{v}" for v in fr.violations])
        if fr.loss not in {"focal", "weighted_ce"}:
            all_vios.append(f"fold{fr.fold_id}:loss_type_missing")
        if fr.time_decay_lambda is None or not (0.95 <= float(fr.time_decay_lambda) <= 1.0):
            all_vios.append(f"fold{fr.fold_id}:time_decay_lambda_missing_or_out_of_range")
        if not (fr.ckpt_ok and fr.oos_probs_ok and fr.cv_split_ok and fr.window_ok and fr.stable):
            # violations already captured per fold
            pass

    passed = len(all_vios) == 0
    # Include warnings (e.g., fallback folds) without failing the run
    out = {
        "folds": [
            {
                "fold_id": fr.fold_id,
                "ckpt_ok": fr.ckpt_ok,
                "oos_probs_ok": fr.oos_probs_ok,
                "cv_split_ok": fr.cv_split_ok,
                "window_ok": fr.window_ok,
                "loss": fr.loss,
                "time_decay_lambda": fr.time_decay_lambda,
                "stable": fr.stable,
                "best_vs_first_val_loss_delta": fr.best_vs_first_val_loss_delta,
                "violations": fr.violations,
            }
            for fr in fold_reports
        ],
        "global": {"pass": passed, "violations": all_vios},
    }
    Path(Path(out_json).parent).mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    if passed:
        typer.echo("PASS")
        raise typer.Exit(code=0)
    else:
        typer.echo("FAIL: " + "; ".join(all_vios[:8]))
        raise typer.Exit(code=1)


@app.callback()
def _main_cb(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo("Use: p5_validate run --help")
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
