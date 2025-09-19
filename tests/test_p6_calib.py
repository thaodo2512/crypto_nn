import json
import numpy as np
import pandas as pd

from cli_p5 import _softmax, _ece_top, calibrate, ensemble, tune_threshold
from typer.testing import CliRunner


def _mk_logits(n=500, n_class=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_class)) * 2.0  # overconfident
    y = rng.integers(0, n_class, size=n)
    # Make them slightly aligned to truth
    z[np.arange(n), y] += 1.0
    return z, y


def test_temp_scaling_ece_down(tmp_path):
    # Create val/oos parquet with logits
    import pyarrow as pa, pyarrow.parquet as pq
    z, y = _mk_logits(600)
    df = pd.DataFrame({"fold_id": [0]*600, "split": ["val"]*300 + ["oos"]*300, "y": np.concatenate([y[:300], y[300:]])})
    for k in range(3):
        df[f"logits_{k}"] = np.concatenate([z[:300, k], z[300:, k]])
    p = tmp_path / "probs.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    runner = CliRunner()
    calib_out = tmp_path / "calib.json"
    r = runner.invoke(calibrate, ["--probs", str(p), "--method", "temperature", "--out", str(calib_out)])
    assert r.exit_code == 0
    cal = json.loads(calib_out.read_text())
    T = cal["0"]["temperature"]
    # Check ECE decreases on val
    val = df[df["split"] == "val"]
    e0 = _ece_top(_softmax(val[["logits_0","logits_1","logits_2"].to_python()] if hasattr(val.columns, 'to_python') else val[["logits_0","logits_1","logits_2"]].to_numpy()), val["y"].to_numpy())
    e1 = _ece_top(_softmax(val[["logits_0","logits_1","logits_2"]].to_numpy(), T=T), val["y"].to_numpy())
    assert e1 <= e0


def test_ev_weight_nonneg_sum1(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    z, y = _mk_logits(400)
    # two folds with different alignment -> different EV
    df = []
    for fid, sl in enumerate([slice(0,200), slice(200,400)]):
        part = pd.DataFrame({"fold_id": [fid]*200, "split": ["oos"]*200, "y": y[sl]})
        for k in range(3):
            part[f"logits_{k}"] = z[sl, k]
        df.append(part)
    df = pd.concat(df, ignore_index=True)
    p = tmp_path / "oos.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    # Calib: identity
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"0": {"temperature": 1.0}, "1": {"temperature": 1.0}}))
    runner = CliRunner()
    out = tmp_path / "ens.json"
    r = runner.invoke(ensemble, ["--calib", str(calib), "--probs", str(p), "--out", str(out)])
    assert r.exit_code == 0
    w = json.loads(out.read_text())["weights"]
    vals = np.array(list(w.values()), dtype=float)
    assert np.all(vals >= 0) and np.isclose(vals.sum(), 1.0)


def test_tau_argmax_ev(tmp_path):
    # Simple synthetic probabilities favor tau around 0.6
    y = np.array([1,1,1,2,2,2,0,0,0])
    P = np.array([
        [0.2,0.7,0.1],
        [0.2,0.6,0.2],
        [0.2,0.55,0.25],
        [0.2,0.1,0.7],
        [0.2,0.2,0.6],
        [0.2,0.25,0.55],
        [0.7,0.15,0.15],
        [0.6,0.2,0.2],
        [0.55,0.25,0.2],
    ])
    import pyarrow as pa, pyarrow.parquet as pq
    df = pd.DataFrame({"fold_id": [0]*len(y), "split": ["oos"]*len(y), "y": y, "p_0": P[:,0], "p_1": P[:,1], "p_2": P[:,2]})
    p = tmp_path / "oos.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"0": {"temperature": 1.0}}))
    ens = tmp_path / "ens.json"
    CliRunner().invoke(ensemble, ["--calib", str(calib), "--probs", str(p), "--out", str(ens)])
    out = tmp_path / "summary.json"
    r = CliRunner().invoke(tune_threshold, ["--ensemble", str(ens), "--probs", str(p), "--grid", "0.50:0.80:0.025", "--cost", "bps:0", "--out", str(out)])
    assert r.exit_code == 0
    summ = json.loads(out.read_text())
    best_tau = float(summ["best_tau"])
    # Expected best around 0.6–0.65 by construction
    assert 0.575 <= best_tau <= 0.675

