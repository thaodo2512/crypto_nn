from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import duckdb
import typer


app = typer.Typer(help="P6 Validator – Validate calibration, ensemble, and threshold tuning artifacts")


def _exists(p: str) -> bool:
    return Path(p).exists()


def _read_parquet_counts(glob: str) -> Dict[int, Dict[str, int]]:
    con = duckdb.connect()
    try:
        q = f"""
        SELECT fold_id, split, COUNT(*) AS cnt
        FROM read_parquet('{glob}')
        GROUP BY 1,2
        ORDER BY 1,2
        """
        rows = con.execute(q).fetchall()
    finally:
        con.close()
    out: Dict[int, Dict[str, int]] = {}
    for fid, split, cnt in rows:
        d = out.setdefault(int(fid), {})
        d[str(split)] = int(cnt)
    return out


@app.command("run")
def run(
    val_oos_glob: str = typer.Option("artifacts/p5_oos_probs/fold*.parquet", "--val-oos-glob"),
    calib_json: str = typer.Option("models/calib.json", "--calib"),
    ensemble_json: str = typer.Option("models/ensemble_5m.json", "--ensemble"),
    summary_json: str = typer.Option("reports/p6_oos_summary.json", "--summary"),
    curves_png: str = typer.Option("reports/p6_curves.png", "--curves"),
    out_json: str = typer.Option("reports/p6_validate.json", "--out-json"),
) -> None:
    violations = []

    # Check presence of files
    if not _exists(calib_json):
        violations.append("calib_json_missing")
    if not _exists(ensemble_json):
        violations.append("ensemble_json_missing")
    if not _exists(summary_json):
        violations.append("summary_json_missing")

    # Read counts for VAL and OOS
    split_counts = _read_parquet_counts(val_oos_glob)
    if not split_counts:
        violations.append("no_val_oos_parquet_found")

    # Load calibration and ensemble
    calib: Dict[str, Dict[str, float]] = {}
    if _exists(calib_json):
        with open(calib_json, "r") as f:
            calib = json.load(f)
    ens: Dict[str, object] = {}
    if _exists(ensemble_json):
        with open(ensemble_json, "r") as f:
            ens = json.load(f)
    weights = ens.get("weights", {}) if isinstance(ens, dict) else {}
    calmap = calib if isinstance(calib, dict) else {}

    # Validate weights
    if weights:
        wsum = sum(float(v) for v in weights.values())
        nonneg = all(float(v) >= 0.0 for v in weights.values())
        if not (0.99 <= wsum <= 1.01) or not nonneg:
            violations.append("ensemble_weights_invalid")
    else:
        violations.append("ensemble_weights_missing")

    # Validate calibration entries exist for folds with VAL data
    for fid, cnts in split_counts.items():
        if cnts.get("val", 0) <= 0:
            violations.append(f"fold{fid}:val_missing")
        if cnts.get("oos", 0) <= 0:
            violations.append(f"fold{fid}:oos_missing")
        if str(fid) not in calmap:
            violations.append(f"fold{fid}:calibration_missing")
        else:
            t = float(calmap[str(fid)].get("temperature", 0.0))
            if not (t > 0):
                violations.append(f"fold{fid}:temperature_invalid")

    # Summary JSON basic checks
    summary_ok = False
    summary: Dict[str, object] = {}
    if _exists(summary_json):
        with open(summary_json, "r") as f:
            summary = json.load(f)
        tau = summary.get("best_tau")
        ev = summary.get("ev_trade")
        if tau is not None and ev is not None:
            summary_ok = True
        else:
            violations.append("summary_fields_missing")

    results = {
        "files": {
            "calib": _exists(calib_json),
            "ensemble": _exists(ensemble_json),
            "summary": _exists(summary_json),
            "curves": _exists(curves_png),
        },
        "split_counts": split_counts,
        "weights": weights,
        "pass": len(violations) == 0,
        "violations": violations,
    }
    Path(Path(out_json).parent).mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    if results["pass"]:
        typer.echo("PASS")
        raise typer.Exit(code=0)
    else:
        typer.echo("FAIL: " + "; ".join(violations))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

