from __future__ import annotations

import duckdb
import typer


app = typer.Typer(help="P1 5m – DuckDB view helper")


def _create_view(db: str, glob: str, view: str) -> None:
    con = duckdb.connect(database=db)
    try:
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{glob}')")
        res = con.execute(f"SELECT COUNT(*) as rows, MIN(ts) as min_ts, MAX(ts) as max_ts FROM {view}").fetchone()
        rows, min_ts, max_ts = res
        typer.echo(f"View '{view}' created in {db}. Rows={rows}, MIN(ts)={min_ts}, MAX(ts)={max_ts}")
    finally:
        con.close()


@app.callback()
def main(
    ctx: typer.Context,
    db: str = typer.Option("meta/duckdb/p1.duckdb", "--db"),
    glob: str = typer.Option("data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet", "--glob"),
    view: str = typer.Option("bars_5m", "--view"),
) -> None:
    # If no subcommand is provided, perform create with given options
    if ctx.invoked_subcommand is None:
        _create_view(db, glob, view)


@app.command()
def create(
    db: str = typer.Option("meta/duckdb/p1.duckdb", "--db"),
    glob: str = typer.Option("data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet", "--glob"),
    view: str = typer.Option("bars_5m", "--view"),
) -> None:
    _create_view(db, glob, view)


if __name__ == "__main__":
    app()
