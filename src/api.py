import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.predict import PredictionError, SentimentPredictor

predictor = SentimentPredictor(settings.model_path)
interface_path = Path(__file__).with_name("static") / "index.html"
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    app.state.predictor = predictor
    yield


app = FastAPI(
    title=settings.app_name,
    description="Classify movie-review sentiment using the trained TF-IDF pipeline.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def interface():
    return FileResponse(interface_path)

@app.middleware("http")
async def log_request(request: Request, call_next):
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )


class ReviewRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=settings.max_review_length,
        description="Movie review text to classify",
    )

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class SentimentResponse(BaseModel):
    text: str = Field(description="Validated review text")
    sentiment: Literal["positive", "negative"] = Field(
        description="Predicted sentiment label"
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Predicted probability of the returned sentiment label",
    )


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Prediction service is temporarily unavailable"},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check model readiness",
    responses={503: {"model": HealthResponse, "description": "Model unavailable"}},
)
def health(response: Response):
    model_loaded = predictor.is_loaded
    if not model_loaded:
        response.status_code = 503

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.post(
    "/predict",
    response_model=SentimentResponse,
    summary="Predict movie-review sentiment",
    responses={503: {"model": ErrorResponse, "description": "Prediction unavailable"}},
)
def predict(review: ReviewRequest, request: Request):
    return request.app.state.predictor.predict(review.text)
