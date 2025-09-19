from __future__ import annotations

import duckdb
import typer


app = typer.Typer(help="DuckDB view helper for P2 features")


@app.command()
def create_view(
    glob: str = typer.Option("data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet", "--glob"),
    view: str = typer.Option("feat_5m", "--view"),
    db: str = typer.Option("meta/duckdb/p1.duckdb", "--db"),
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

