.PHONY: p1_5m_ingest p1_5m_qa p1_5m_view

p1_5m_ingest:
	python ingest_cg_5m.py ingest --symbol BTCUSDT --tf 5m --days 180 --out data/parquet/5m/BTCUSDT

p1_5m_qa:
	python qa_p1_5m.py qa --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out reports/p1_qa_core_5m.json

p1_5m_view:
	python duckview.py create --db meta/duckdb/p1.duckdb --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --view bars_5m

