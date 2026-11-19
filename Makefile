# Shortcuts. Windows users: run the command after each colon line directly.
.PHONY: setup data train up down test lint baseline drift report

setup:      ## install everything + enable the commit hook
	pip install -r requirements-dev.txt
	git config core.hooksPath .githooks

data:       ## validate raw data and make train/test split
	python -m fraud.data.preprocess

train:      ## train, track and register (needs `make up` first)
	python -m fraud.models.train

up:         ## start MLflow, API, Prometheus, Grafana
	docker compose up -d --build

down:
	docker compose down

test:
	pytest -q

lint:
	ruff check .

baseline:   ## send the clean test set to the API
	python -m fraud.monitoring.send_to_api data/processed/test.csv

drift:      ## make drifted datasets and send them to the API
	python -m fraud.monitoring.simulate_drift
	for f in data/drift/*.csv; do python -m fraud.monitoring.send_to_api $$f; done

report:     ## Evidently drift report -> reports/drift_report.html
	python -m fraud.monitoring.drift_report
