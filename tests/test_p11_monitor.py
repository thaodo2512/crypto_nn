import json
import numpy as np
import pandas as pd

from monitor_p11 import ks_two_sample, cusum_regimes


def test_ks_drift_flags():
    x = np.random.randn(1000)
    y = np.random.randn(1000) + 0.5  # shifted
    d, p = ks_two_sample(x, y)
    assert p < 0.05


def test_cusum_segments():
    # Build a series with a mean shift
    s1 = np.random.randn(500)
    s2 = np.random.randn(500) + 1.0
    idx = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
    s = pd.Series(np.concatenate([s1, s2]), index=idx)
    r = cusum_regimes(s, boot_n=100, q=0.95)
    # Expect at least one change flagged
    assert any(item.get("changed") for item in r["series"])


def test_trigger_written(tmp_path, monkeypatch):
    # Write drift & regimes reports
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    drift = {"alert_rate_chart": {"out_of_control": True}}
    regimes = {"series": []}
    (tmp_path / "reports" / "p11_drift.json").write_text(json.dumps(drift))
    (tmp_path / "reports" / "p11_regimes.json").write_text(json.dumps(regimes))
    # Run retrain command
    from monitor_p11 import retrain
    from typer.testing import CliRunner
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        out = tmp_path / "ops" / "retrain_trigger.json"
        r = CliRunner().invoke(retrain, ["--config", "conf/retrain.yaml", "--out", str(out)])
        assert r.exit_code == 0
        assert out.exists()
    finally:
        os.chdir(cwd)

