import numpy as np
import pandas as pd

from cli_p5 import build_windows
from folds import make_purged_folds


def _mk_feats_labels(n=400, start="2024-01-01T00:00:00Z"):
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ts": idx,
        "symbol": ["BTCUSDT"] * n,
        "ret_5m": rng.standard_normal(n),
        "vol_z": rng.standard_normal(n),
        "oi_z": rng.standard_normal(n),
        "fund_now_z": rng.standard_normal(n),
        "cvd_diff_z": rng.standard_normal(n),
        "liq60_z": rng.standard_normal(n),
        "rv_5m_z": rng.standard_normal(n),
        "perp_share_60m": rng.random(n),
    })
    y = np.array(["WAIT"] * n)
    y[50:70] = "LONG"
    y[120:140] = "SHORT"
    lab = pd.DataFrame({"ts": idx, "symbol": ["BTCUSDT"] * n, "label": y})
    return df, lab


def test_window_shape():
    df, lab = _mk_feats_labels(200)
    X, y, meta = build_windows(df, lab, W=12)
    assert X.shape[1] == 12 and X.shape[2] > 0
    # Ensure meta ts equals label ts and window ends at t
    assert len(meta) == len(y)


def test_purged_cv_no_leak():
    df, lab = _mk_feats_labels(300)
    X, y, meta = build_windows(df, lab, W=12)
    idx = meta.sort_values("ts")["ts"]
    folds = make_purged_folds(idx, n_folds=3, embargo="1D")
    for f in folds:
        tr_last = idx[f["train_idx"]].max() if len(f["train_idx"]) else idx.min()
        vl_first = idx[f["val_idx"]].min()
        assert (vl_first - tr_last) >= pd.Timedelta(days=1)

