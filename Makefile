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

.PHONY: p4_sampling
p4_sampling:
	python cli_p4.py iforest-train \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --out data/masks/ifgate_5m.parquet --q 0.995 --rolling-days 30 --seed 42
	python cli_p4.py smote-windows \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --mask data/masks/ifgate_5m.parquet --W 144 --out data/aug/train_smote --seed 42
	python cli_p4.py report-classmix \
	  --pre data/train/ --post data/aug/train_smote/ --out reports/p4_classmix.json

.PHONY: p5_train
p5_train:
	python cli_p5.py train --model gru --window 144 --cv walkforward --embargo 1D \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --out "models/gru_5m" --seed 42 --folds 5
