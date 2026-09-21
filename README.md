# Sentiment Analysis Microservice

A five-day, interview-focused ML engineering project for classifying IMDB movie reviews. It covers data validation, model training and comparison, experiment tracking, API serving, automated tests, and container preparation.

## Current model and results

- Training data: a reproducible sample of 10,000 IMDB reviews, with duplicate text removed before splitting.
- Features: TF-IDF and the classifier are saved together as one scikit-learn pipeline.
- Selection: four classifiers compared using five-fold stratified cross-validation and macro F1.
- Selected classifier: SGD with log loss.
- Mean cross-validation macro F1: **86.91%**.
- Final held-out test accuracy: **86.08%** on 1,997 reviews.
- Error analysis: 155 false positives and 123 false negatives in the held-out test set.

MLflow records candidate comparisons, final metrics, and the selected pipeline. The local error report is written to `reports/error_analysis.csv`.

## Project structure

```text
src/train.py                 Data loading, validation, comparison, training, MLflow
src/config.py                API configuration
src/predict.py               Model loading and prediction service
src/api.py                   FastAPI validation, endpoints, and request logging
src/app.py                   Optional Streamlit frontend
tests/                       Pytest data, prediction, and API tests
models/pipeline.joblib       Selected trained pipeline
.github/workflows/tests.yml  GitHub Actions test workflow
Dockerfile                   FastAPI container image
```

## Setup on Windows PowerShell

From the project root:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe -m pytest -q tests
```

To retrain and compare models:

```powershell
& .\.venv\Scripts\python.exe src\train.py
```

Training may download IMDB from Hugging Face; if it is unavailable, the datasets library may use its local cache. The script creates `data/` for its sample CSV, logs experiments to MLflow, and saves the selected pipeline in `models/`.

## Run the API

```powershell
& .\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try `POST /predict` or `GET /health`. A prediction request looks like:

```json
{"text": "This movie was excellent."}
```

The API strips surrounding whitespace, rejects invalid input with HTTP 422, and returns a sentiment label and estimated confidence. It logs request path, status, and duration without logging review text.

The optional frontend runs separately:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run src\app.py
```

## Docker

The Docker image runs the FastAPI service. It copies the selected `models/pipeline.joblib`; commit that artifact with the application before building from a fresh checkout. Other generated model files, datasets, MLflow runs, and reports are ignored by Git.

```powershell
docker build -t sentiment-service .
docker run --rm -p 8000:8000 sentiment-service
```

Then open `http://localhost:8000/docs`. Docker is not installed in the current Windows environment, so an image build and container run have not yet been verified here.

## CI and next production steps

`.github/workflows/tests.yml` runs pytest on pushes and pull requests, and both have passed on GitHub. Container runtime verification, deployment, and persistent monitoring remain to be completed.
