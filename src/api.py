from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")


class ReviewRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=SentimentResponse)
def predict(review: ReviewRequest):
    result = predict_sentiment(review.text)
    return result
