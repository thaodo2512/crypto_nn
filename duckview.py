from __future__ import annotations

import duckdb
import typer


app = typer.Typer(help="P1 5m – DuckDB view helper")


@app.command()
def create(
    db: str = typer.Option("meta/duckdb/p1.duckdb", "--db"),
    glob: str = typer.Option("data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet", "--glob"),
    view: str = typer.Option("bars_5m", "--view"),
) -> None:
    con = duckdb.connect(database=db)
    try:
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{glob}')")
        res = con.execute(f"SELECT COUNT(*) as rows, MIN(ts) as min_ts, MAX(ts) as max_ts FROM {view}").fetchone()
        rows, min_ts, max_ts = res
        typer.echo(f"View '{view}' created in {db}. Rows={rows}, MIN(ts)={min_ts}, MAX(ts)={max_ts}")
    finally:
        con.close()


if __name__ == "__main__":
    app()

