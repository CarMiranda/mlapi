![Build](https://github.com/CarMiranda/mlapi/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/CarMiranda/mlapi/branch/main/graph/badge.svg?token=VATZ7Z8DOW)](https://codecov.io/gh/CarMiranda/mlapi)

This repository allows to train a Random Forest classifier on census data, with a salary class as target.
The model is packaged and deployed to an HTTP API running on [Heroku](https://mlapi-cm.herokuapp.com/), which allows to run inferences.
One can use `dvc repro` to run data preparation and model training.

### Data processing
Processing consists of removing all whitespaces from the raw data.
One can run:
```python
   python starter/clean_data.py -i data/census.csv -o data/census_clean.csv
```
or simply : `dvc repro clean`

This step output `data/census_clean.csv`, which is tracked by DVC.

### Training
A one-hot encoder is used to encode categorical data, and a label binarizer transforms the target _salary_ variable.  
A Random Forest classifier is trained using 5-fold cross-validation.
One can run:
```python
   python starter/train_model.py -i data/census_clean.csv -m model -M metrics
```
or simply : `dvc repro train`

This step outputs a `model.pkl` (an sklearn pipeline made up of a column transformer and a random forest), a `label_encoder.pkl` (label binarizer), and two sets of metrics (`metrics/train_metrics.json` and `metrics/test_metrics.json`) which are average+std metrics given by 5-fold cross-validation. Models are tracked and cached by dvc, while metrics are not cached.

### Evaluation
Evaluation can be done using "global" metrics or rather metrics computed on slices of data.
One can run:
```python
   python starter/evaluate.py -i data/census_clean.csv -m model [-k KEY [-v VAL]] [-M METRICS_DIR]
```
where:
- `KEY` is the name of a column in the training dataset. If `KEY` is set, but `VAL` is not, evaluation is run for every modality of `KEY` column.
- `VAL` is a modality of the given column.
If neither `KEY`, nor `VAL` are specified, the script yields global metrics.

For example,
```python
   python starter/evaluate.py -i data/census_clean.csv -m model -k sex -M metrics
```
returns something similar to:

Value,Precision,Recall,FBeta  
Male,0.95474,0.92765,0.94100  
Female,0.95183,0.90500,0.92783  

and writes metrics results to json files in the `metrics` folder (see example `metrics/sex_Female_metrics.json` and `metrics/sex_Male_metrics.json`).

### Testing
In order to run the project's tests, one can run:
```python
   coverage run
   coverage report
```
or `pytest`.
