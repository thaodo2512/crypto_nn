import numpy as np
import pandas as pd
from typer.testing import CliRunner

from app.explain.cli import app as explain_app


def test_if_shap_present_only_on_alert(tmp_path):
    # Build a fake window
    W, F = 144, 5
    win = np.random.randn(W, F).astype(np.float32)
    np.save(tmp_path / "w.npy", win)
    # Fake ckpt/onnx by exporting a tiny model
    import torch
    from cli_p5 import GRUClassifier
    from app.export.onnx_export import export_fp16_probs
    net = GRUClassifier(input_dim=F, hidden=8)
    ckpt = tmp_path / "b.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    onnx = tmp_path / "m.onnx"
    export_fp16_probs(str(ckpt), str(onnx), temperature=1.0)
    # IF csv with alert for id1, non-alert for id2
    df = pd.DataFrame({"id": ["id1", "id2"], "alert": [1, 0]})
    csv = tmp_path / "alerts.csv"
    df.to_csv(csv, index=False)
    # id1 should include if_shap
    out1 = tmp_path / "explain1"
    r1 = CliRunner().invoke(explain_app, [
        "run", "--decision-id", "id1",
        "--window-npy", str(tmp_path/"w.npy"),
        "--ckpt", str(ckpt), "--onnx", str(onnx),
        "--if-csv", str(csv), "--out-dir", str(out1),
    ])
    assert r1.exit_code == 0, r1.output
    data1 = (out1/"id1.json").read_text()
    assert "if_shap" in data1
    # id2 should not include if_shap
    out2 = tmp_path / "explain2"
    r2 = CliRunner().invoke(explain_app, [
        "run", "--decision-id", "id2",
        "--window-npy", str(tmp_path/"w.npy"),
        "--ckpt", str(ckpt), "--onnx", str(onnx),
        "--if-csv", str(csv), "--out-dir", str(out2),
    ])
    assert r2.exit_code == 0, r2.output
    data2 = (out2/"id2.json").read_text()
    assert "if_shap" not in data2

