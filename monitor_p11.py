from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import mlflow
import numpy as np
import pandas as pd
import typer


app = typer.Typer(help="P11 – Monitoring: drift detection, regimes, auto-retrain triggers")


def _read_parquet(glob: str, sel: str = "*") -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def ks_two_sample(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.sort(x)
    y = np.sort(y)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return 0.0, 1.0
    data_all = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, data_all, side="right") / nx
    cdf_y = np.searchsorted(y, data_all, side="right") / ny
    d = np.max(np.abs(cdf_x - cdf_y))
    ne = nx * ny / (nx + ny)
    # Asymptotic p-value (Kolmogorov distribution)
    p = 2.0 * np.exp(-2.0 * ne * d * d)
    p = float(np.clip(p, 0.0, 1.0))
    return float(d), p


def control_chart(alert_rate: pd.Series) -> Dict[str, float]:
    mu = float(alert_rate.mean())
    sigma = float(alert_rate.std(ddof=1)) if len(alert_rate) > 1 else 0.0
    upper = mu + 3 * sigma
    lower = max(0.0, mu - 3 * sigma)
    last = float(alert_rate.iloc[-1]) if len(alert_rate) else 0.0
    return {"mu": mu, "sigma": sigma, "ucl": upper, "lcl": lower, "last": last, "out_of_control": last > upper or last < lower}


def cusum_regimes(rv: pd.Series, boot_n: int = 200, q: float = 0.99) -> Dict:
    z = (rv - rv.mean()) / (rv.std(ddof=1) or 1.0)
    k = 0.25  # allowance
    # Bootstrap threshold from first 1000 samples or full series
    ref = z.iloc[: min(1000, len(z))].to_numpy()
    def max_cusum(arr: np.ndarray) -> float:
        s_pos = 0.0
        s_neg = 0.0
        m = 0.0
        for v in arr:
            s_pos = max(0.0, s_pos + v - k)
            s_neg = min(0.0, s_neg + v + k)
            m = max(m, s_pos, -s_neg)
        return m
    boots = [max_cusum(np.random.choice(ref, size=len(ref), replace=True)) for _ in range(boot_n)]
    h = float(np.quantile(boots, q))
    # Run CUSUM and mark regime changes
    s_pos = 0.0
    s_neg = 0.0
    regime_id = 0
    regimes: List[Dict] = []
    for ts, v in z.items():
        s_pos = max(0.0, s_pos + float(v) - k)
        s_neg = min(0.0, s_neg + float(v) + k)
        changed = False
        if s_pos > h:
            regime_id += 1
            s_pos = 0.0
            s_neg = 0.0
            changed = True
        if -s_neg > h:
            regime_id += 1
            s_pos = 0.0
            s_neg = 0.0
            changed = True
        regimes.append({"ts": ts.isoformat(), "regime_id": regime_id, "changed": changed})
    return {"threshold": h, "k": k, "q": q, "series": regimes}


@app.command("run")
def run(
    features: str = typer.Option(..., "--features"),
    decisions: str = typer.Option(..., "--decisions"),
    out: str = typer.Option("reports", "--out"),
    mlflow_experiment: str = typer.Option("btc_5m", "--mlflow-experiment"),
) -> None:
    feat = _read_parquet(features)
    dec = _read_parquet(decisions)
    # Impute ratios
    impute_f = float(feat.get("_imputed_funding_now", pd.Series(dtype=int)).mean() or 0.0)
    impute_oi = float(feat.get("_imputed_oi_now", pd.Series(dtype=int)).mean() or 0.0)
    # Class mix and alert rate per day
    dec["date"] = dec["ts"].dt.date
    mix = dec["side"].value_counts(normalize=True).to_dict() if "side" in dec else {}
    daily = dec.groupby("date").apply(lambda d: (d.get("side", pd.Series()).ne("WAIT")).mean() if "side" in d else 0.0).astype(float)
    cc = control_chart(daily)
    ev_trade = float(dec.get("EV", pd.Series(dtype=float)).mean() or 0.0)
    # Persist drift report
    Path(out).mkdir(parents=True, exist_ok=True)
    drift = {"impute_funding": impute_f, "impute_oi": impute_oi, "class_mix": mix, "alert_rate_chart": cc, "ev_trade": ev_trade}
    with open(Path(out) / "p11_drift.json", "w") as f:
        json.dump(drift, f, indent=2)
    # MLflow log
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"monitor_{int(time.time())}"):
        mlflow.log_metrics({"impute_funding": impute_f, "impute_oi": impute_oi, "ev_trade": ev_trade, "alert_rate": cc.get("last", 0.0)})
        for k, v in mix.items():
            mlflow.log_metric(f"mix_{k}", float(v))
        mlflow.set_tags({"phase": "p11"})
    typer.echo("Monitoring report written.")


@app.command("regimes")
def regimes(rv: str = typer.Option(..., "--rv"), out: str = typer.Option("reports/p11_regimes.json", "--out")) -> None:
    df = _read_parquet(rv)
    # Expect rv_5m or rv_5m_z
    if "rv_5m" in df:
        s = df.sort_values("ts")["rv_5m"].astype(float)
    elif "rv_5m_z" in df:
        s = df.sort_values("ts")["rv_5m_z"].astype(float)
    else:
        raise typer.BadParameter("rv parquet must contain rv_5m or rv_5m_z")
    r = cusum_regimes(s)
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    typer.echo("Regimes report written.")


@app.command("retrain")
def retrain(config: str = typer.Option(..., "--config"), out: str = typer.Option("ops/retrain_trigger.json", "--out")) -> None:
    # Minimal retrain trigger: if latest regimes changed or drift out-of-control
    # Read previous reports
    drift = json.loads(Path("reports/p11_drift.json").read_text()) if Path("reports/p11_drift.json").exists() else {"alert_rate_chart": {"out_of_control": False}}
    regimes = json.loads(Path("reports/p11_regimes.json").read_text()) if Path("reports/p11_regimes.json").exists() else {"series": []}
    changed = any(item.get("changed") for item in regimes.get("series", [])[-10:])
    ooc = drift.get("alert_rate_chart", {}).get("out_of_control", False)
    if changed or ooc:
        Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
        payload = {"ts": int(time.time()), "reason": "regime_shift" if changed else "alert_rate_ooc", "config": config}
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        typer.echo("Retrain trigger written.")
    else:
        typer.echo("No retrain trigger conditions met.")


if __name__ == "__main__":
    app()

