.PHONY: p1_5m_ingest p1_5m_qa p1_5m_view

p1_5m_ingest:
	python ingest_cg_5m.py ingest --symbol BTCUSDT --tf 5m --days 180 --out data/parquet/5m/BTCUSDT

p1_5m_qa:
	python qa_p1_5m.py qa --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out reports/p1_qa_core_5m.json

p1_5m_view:
	python duckview.py create --db meta/duckdb/p1.duckdb --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --view bars_5m

.PHONY: p3_label
p3_label:
	python label_p3.py triple-barrier --tf 5m --k 1.2 --H 36 \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --out "data/labels/5m/BTCUSDT"
	python label_p3.py validate \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --report "reports/p3_qa_5m_80d.json"
