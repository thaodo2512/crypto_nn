import numpy as np
import pandas as pd

from policy_p7 import _ev_decision_row, _dynamic_k, decide
from typer.testing import CliRunner


def test_ev_gate_wait_when_negative():
    # Very low probs → EV <= 0
    side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
        p_long=0.1, p_short=0.1, close=100.0, atr_pct=0.01, k_min=1.0, k_max=1.5, vol_pct=0.5, cost=type("C", (), {"cost_frac": lambda self: 0.01})()
    )
    assert side == "WAIT" and reason == "ev<=0"


def test_no_reentry_within_H(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    # Two consecutive timestamps with high prob; cooldown should prevent second entry
    idx = pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC")
    probs = pd.DataFrame({"ts": idx, "p_0": [0.0, 0.0], "p_1": [0.9, 0.9], "p_2": [0.05, 0.05]})
    atr = pd.DataFrame({"ts": idx, "close": [100, 100.5], "atr_pct": [0.01, 0.011], "vol_pctile": [0.5, 0.5]})
    pfile = tmp_path / "probs.parquet"
    afile = tmp_path / "atr.parquet"
    pq.write_table(pa.Table.from_pandas(probs, preserve_index=False), pfile)
    pq.write_table(pa.Table.from_pandas(atr, preserve_index=False), afile)
    out = tmp_path / "dec"
    r = CliRunner().invoke(decide, ["--probs", str(pfile), "--atr", str(afile), "--k-range-min", "1.0", "--k-range-max", "1.5", "--H", "36", "--out", str(out)])
    assert r.exit_code == 0
    # Read back and check cooldown
    import duckdb
    con = duckdb.connect()
    try:
        df = con.execute(f"select * from read_parquet('{str(out)}/**/*.parquet') order by ts").df()
    finally:
        con.close()
    assert df.iloc[0]["side"] != "WAIT" and df.iloc[1]["reason"] == "cooldown"


def test_dynamic_k_mapping():
    k_low = _dynamic_k(0.0, 1.0, 1.5)
    k_mid = _dynamic_k(0.5, 1.0, 1.5)
    k_hi = _dynamic_k(1.0, 1.0, 1.5)
    assert 1.0 <= k_low < k_mid < k_hi <= 1.5

