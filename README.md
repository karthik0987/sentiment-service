# Sentiment Analysis Microservice

End-to-end ML pipeline for movie review sentiment classification. Built to go beyond Jupyter notebooks — this project covers data ingestion, model training with experiment tracking, a REST API for inference, and a web frontend, all containerized with Docker.

## What it does

You type in a movie review → the app tells you if it's positive or negative, along with a confidence score.

Under the hood:
- Pulls 10k IMDB reviews from HuggingFace Datasets (no static CSVs)
- Trains a TF-IDF + Logistic Regression classifier (~87% accuracy)
- Logs all experiments (params, metrics, model artifacts) to MLflow
- Serves predictions through a FastAPI REST API
- Streamlit frontend that talks to the API in real time

## Tech Stack

- **ML/NLP:** scikit-learn, TF-IDF vectorization, Logistic Regression
- **Data:** HuggingFace Datasets API
- **Experiment Tracking:** MLflow
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Containerization:** Docker

## Project Structure

```
sentiment-service/
├── src/
│   ├── train.py       # training pipeline + MLflow logging
│   ├── predict.py     # inference module
│   ├── api.py         # FastAPI endpoints
│   └── app.py         # Streamlit UI
├── models/            # saved model + vectorizer (.joblib)
├── data/              # cached dataset
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

```bash
# clone and enter the project
git clone https://github.com/yourusername/sentiment-service.git
cd sentiment-service

# create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# train the model (downloads data + logs to MLflow)
python src/train.py
```

## Running

You need two terminals:

**Terminal 1 — API server:**
```bash
uvicorn src.api:app --reload --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run src/app.py
```

Then open `http://localhost:8501` in your browser.

## API Usage

Health check:
```bash
curl http://localhost:8000/health
```

Predict sentiment:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic"}'
```

Response:
```json
{
  "text": "This movie was absolutely fantastic",
  "sentiment": "positive",
  "confidence": 0.9312
}
```

Interactive docs available at `http://localhost:8000/docs`.

## MLflow Dashboard

After training, you can inspect logged experiments:

```bash
mlflow ui
```

Opens at `http://localhost:5000`. Shows all runs with hyperparameters, accuracy, precision, recall, F1, and saved model artifacts.

## Docker

```bash
docker build -t sentiment-service .
docker run -p 8000:8000 -p 8501:8501 sentiment-service
```

## Things I'd improve with more time

- Swap Logistic Regression for a fine-tuned DistilBERT (would push accuracy to ~93%)
- Add CI/CD with GitHub Actions to retrain on new data
- Set up model versioning with MLflow Model Registry
- Add batch prediction endpoint for bulk reviews
- Deploy on AWS (EC2 or ECS) with a proper domain

## License

MIT
