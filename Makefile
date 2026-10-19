.PHONY: setup test demo-data demo api validate deploy run

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt

test:
	pytest -q

demo-data:
	python src/ingestion/download_data.py --source synthetic

demo: demo-data
	PYTHONPATH=src python -m demand.pipeline --raw data/raw --out outputs --fast

api:
	uvicorn src.api.main:app --reload

validate:
	databricks bundle validate -t dev

deploy:
	databricks bundle deploy -t dev

run:
	databricks bundle run demand_forecasting_job -t dev
