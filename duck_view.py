from __future__ import annotations

import duckdb
import typer


app = typer.Typer(help="DuckDB view helper for P2 features")


def _create(db: str, glob: str, view: str) -> None:
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
    glob: str = typer.Option("data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet", "--glob"),
    view: str = typer.Option("feat_5m", "--view"),
    db: str = typer.Option("meta/duckdb/p1.duckdb", "--db"),
) -> None:
    if ctx.invoked_subcommand is None:
        _create(db, glob, view)


@app.command()
def create_view(
    glob: str = typer.Option("data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet", "--glob"),
    view: str = typer.Option("feat_5m", "--view"),
    db: str = typer.Option("meta/duckdb/p1.duckdb", "--db"),
) -> None:
    _create(db, glob, view)


if __name__ == "__main__":
    app()
