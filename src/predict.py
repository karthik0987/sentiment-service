from pathlib import Path

import joblib


class PredictionError(RuntimeError):
    """Raised when the loaded model cannot produce a prediction."""


class SentimentPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model from {self.model_path}"
            ) from exc

        required_methods = ("predict", "predict_proba")
        missing_methods = [
            method for method in required_methods if not hasattr(pipeline, method)
        ]
        if missing_methods:
            raise TypeError(
                "Loaded model is missing required methods: "
                + ", ".join(missing_methods)
            )

        self.pipeline = pipeline

    def predict(self, text: str) -> dict:
        if not self.is_loaded:
            raise PredictionError("The sentiment model has not been loaded")

        try:
            prediction = self.pipeline.predict([text])[0]
            probability = self.pipeline.predict_proba([text])[0]
        except Exception as exc:
            raise PredictionError("The model could not produce a prediction") from exc

        return {
            "text": text,
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": round(float(max(probability)), 4),
        }
