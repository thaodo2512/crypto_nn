from __future__ import annotations

import time
from pathlib import Path

import duckdb
import typer


app = typer.Typer(help="P2 bench – measure ms per bar for reading features")


def read_count(glob: str) -> int:
    con = duckdb.connect()
    try:
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob}')").fetchone()[0]
    finally:
        con.close()
    return int(rows)


@app.command()
def bench(
    glob: str = typer.Option(..., "--glob"),
    report: str = typer.Option(..., "--report"),
) -> None:
    t0 = time.perf_counter()
    rows = read_count(glob)
    dt = time.perf_counter() - t0
    ms_per_bar = (dt * 1000.0 / rows) if rows else 0.0
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        f.write("rows,seconds,ms_per_bar\n")
        f.write(f"{rows},{dt:.6f},{ms_per_bar:.6f}\n")
    if ms_per_bar > 50.0:
        typer.echo(f"Benchmark failed: {ms_per_bar:.3f} ms/bar > 50")
        raise typer.Exit(code=1)
    typer.echo(f"Benchmark: {rows} rows in {dt:.3f}s ⇒ {ms_per_bar:.3f} ms/bar")


if __name__ == "__main__":
    app()

