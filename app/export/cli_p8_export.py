from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import typer

from .onnx_export import export_fp16_probs, ExportSpec
from .parity import ParityReport, parity_check, sha256_path


app = typer.Typer(help="P8 – Export ONNX FP16 and validate parity vs calibrated PyTorch (window=144)")


def _read_ensemble(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"Ensemble file missing: {path}")
    d = json.loads(p.read_text())
    # Expected keys
    if "temperature" not in d:
        raise typer.BadParameter("Ensemble JSON must include 'temperature'")
    w = d.get("weights", {}) or {}
    if not w:
        w = {"gru": 1.0}
    if any(float(v) < 0.0 for v in w.values()):
        raise typer.BadParameter("Ensemble weights must be non-negative")
    s = sum(float(v) for v in w.values())
    if s <= 0:
        raise typer.BadParameter("Sum of ensemble weights must be positive")
    # Normalize (safety)
    d["weights"] = {k: float(v) / s for k, v in w.items()}
    return d


@app.command("onnx")
def cli(
    ckpt: str = typer.Option(..., "--ckpt", help="Checkpoint path or glob (first used)"),
    out: str = typer.Option("export/model_5m_fp16.onnx", "--out"),
    window: int = typer.Option(144, "--window"),
    fp16: bool = typer.Option(True, "--fp16/--fp32", help="Export FP16 graph (must be true)"),
    preproc: str = typer.Option(..., "--preproc", help="Preprocessing YAML path (checksum only)"),
    ensemble: str = typer.Option(..., "--ensemble", help="Ensemble JSON with 'temperature' and 'weights'"),
    sample: int = typer.Option(2048, "--sample"),
    skip_export: bool = typer.Option(False, "--skip-export", help="Validate parity only; don't export"),
) -> None:
    if not fp16:
        raise typer.BadParameter("Only FP16 export is allowed in P8")

    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("p8")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/p8_export.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ens = _read_ensemble(ensemble)
    T = float(ens["temperature"])
    spec = ExportSpec(window=window)

    if not skip_export:
        export_fp16_probs(ckpt, out, temperature=T, spec=spec)
        logger.info(f"Exported ONNX to {out} (window={window}, fp16)")

    # Parity
    mse = parity_check(ckpt, out, temperature=T, samples=sample, window=window)
    onnx_sha = sha256_path(out)
    pre_sha = sha256_path(preproc) if Path(preproc).exists() else ""
    ens_sha = sha256_path(ensemble)

    report = ParityReport(
        mse_probs=mse,
        n_samples=sample,
        onnx_sha256=onnx_sha,
        preproc_sha256=pre_sha,
        ensemble_sha256=ens_sha,
        opset=spec.opset,
        window=spec.window,
        dynamic_axes=("batch",),
    )
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/p8_parity.json").write_text(report.to_json())
    logger.info(report.to_json())

    if mse >= 1e-3:
        raise SystemExit(1)
    typer.echo(f"PASS P8 parity MSE={mse:.6g}")


if __name__ == "__main__":
    # Allow both styles:
    #  - python -m app.export.cli_p8_export onnx --ckpt ...
    #  - python -m app.export.cli_p8_export --ckpt ...
    # Some environments pass a literal 'onnx' which our Typer app may not treat
    # as a subcommand; normalize by stripping it if present.
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "onnx":
        sys.argv.pop(1)
    app()
