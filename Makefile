VENV?=.venv
PY?=python

.PHONY: venv install data train infer

venv:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate; pip install -U pip

install:
	. $(VENV)/bin/activate; pip install -r requirements.txt

data:
	. $(VENV)/bin/activate; $(PY) generate_synthetic_data.py --rows 5000 --out_path data/training_data.csv

train:
	. $(VENV)/bin/activate; $(PY) train.py --data_path data/training_data.csv --output_dir artifacts

infer:
	. $(VENV)/bin/activate; $(PY) infer.py --model_path artifacts/model.joblib --input_json sample.json --threshold 0.5 --output_path predictions.json