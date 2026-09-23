# Sentiment Service

A sentiment analysis service for IMDB movie reviews, with reproducible training, model comparison, MLflow experiment tracking, a FastAPI inference API, automated tests, Docker packaging, CI, and a browser interface.

## Live deployment

The FastAPI service is deployed on Render:

- Web application: https://sentiment-service-ugie.onrender.com/
- Interactive API documentation: https://sentiment-service-ugie.onrender.com/docs
- Health check: https://sentiment-service-ugie.onrender.com/health
- Prediction endpoint: `POST https://sentiment-service-ugie.onrender.com/predict`

The deployed service has been verified with a loaded model and a real prediction request.

> Render's free instance can sleep after inactivity. The first request after it sleeps may take 50 seconds or longer while the service starts again.

### Try a prediction

PowerShell:

```powershell
$body = @{
    text = "This movie was excellent and I loved every minute."
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://sentiment-service-ugie.onrender.com/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Request body:

```json
{
  "text": "This movie was excellent and I loved every minute."
}
```

Example response:

```json
{
  "text": "This movie was excellent and I loved every minute.",
  "sentiment": "positive",
  "confidence": 0.9621
}
```

## System flow

```text
Review text
    |
    v
FastAPI input validation
    |
    v
Saved scikit-learn pipeline
    |
    +--> TF-IDF converts text into numeric features
    |
    +--> SGD classifier predicts sentiment
    |
    v
JSON response: sentiment + confidence
```

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
src/static/                  Browser interface assets
    index.html                 Page structure
    styles.css                 Interface styling
    app.js                     Prediction requests and result display
models/pipeline.joblib       Selected trained pipeline
tests/                       Pytest data, prediction, and API tests
.github/workflows/tests.yml  GitHub Actions test workflow
Dockerfile                   FastAPI container image
render.yaml                  Render deployment configuration
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

## Run the API locally

```powershell
& .\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try `POST /predict` or `GET /health`.

The API strips surrounding whitespace, rejects invalid input with HTTP 422, and returns a sentiment label and estimated confidence. It logs request path, status, and duration without logging review text.

## Docker

The Docker image runs the FastAPI service. It copies the selected `models/pipeline.joblib`; commit that artifact with the application before building from a fresh checkout. Other generated model files, datasets, MLflow runs, and reports are ignored by Git.

```powershell
docker build -t sentiment-service .
docker run --detach --rm --name sentiment-service --publish 8000:8000 sentiment-service
```

Then open `http://localhost:8000/docs`. The image build, API startup, model loading, `/health`, and positive and negative predictions have been verified locally with Docker Desktop.

The image includes a Docker health check that calls `/health`. Check its status with:

```powershell
docker inspect sentiment-service --format "{{.State.Health.Status}}"
```

A ready container reports `healthy`. Stop and remove the temporary container with `docker stop sentiment-service`.

## CI, deployment, and monitoring

- `.github/workflows/tests.yml` runs pytest for pushes and pull requests.
- Render builds the Docker image after checks pass on the `main` branch.
- Render calls `/health` to verify that the application and model are ready.
- The API logs request method, path, status, and duration for operational visibility without recording review text.
