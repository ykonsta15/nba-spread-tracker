.PHONY: setup init-db backfill train predict update app test

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

init-db:
	python3 scripts/init_db.py

backfill:
	python3 scripts/backfill_season.py

train:
	python3 scripts/train_model.py

predict:
	python3 scripts/predict_today.py

update:
	python3 scripts/update_results.py

app:
	python3 -m streamlit run src/app/app.py

test:
	python3 -m pytest -q
