import joblib
import pytest

from src.predict import PredictionError, SentimentPredictor


class FakePipeline:
    def predict(self, texts):
        self.received_texts = texts
        return [1]

    def predict_proba(self, texts):
        return [[0.12, 0.88]]


class BrokenPipeline(FakePipeline):
    def predict(self, texts):
        raise ValueError("Internal model failure")


class NegativePipeline(FakePipeline):
    def predict(self, texts):
        return [0]

    def predict_proba(self, texts):
        return [[0.81, 0.19]]


class IncompletePipeline:
    def predict(self, texts):
        return [1]


def test_load_and_predict_with_raw_text(tmp_path):
    model_path = tmp_path / "fake_pipeline.joblib"
    joblib.dump(FakePipeline(), model_path)
    predictor = SentimentPredictor(model_path)

    assert predictor.is_loaded is False
    predictor.load()
    assert predictor.is_loaded is True

    result = predictor.predict("A wonderful movie")
    assert predictor.pipeline.received_texts == ["A wonderful movie"]
    assert result == {
        "text": "A wonderful movie",
        "sentiment": "positive",
        "confidence": 0.88,
    }


def test_missing_model_file_is_rejected(tmp_path):
    predictor = SentimentPredictor(tmp_path / "missing.joblib")

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        predictor.load()
    assert predictor.is_loaded is False


def test_negative_label_returns_negative_sentiment(tmp_path):
    model_path = tmp_path / "negative_pipeline.joblib"
    joblib.dump(NegativePipeline(), model_path)
    predictor = SentimentPredictor(model_path)
    predictor.load()

    result = predictor.predict("A dull movie")
    assert result["sentiment"] == "negative"
    assert result["confidence"] == 0.81


def test_predict_before_loading_is_rejected(tmp_path):
    predictor = SentimentPredictor(tmp_path / "fake_pipeline.joblib")

    with pytest.raises(RuntimeError, match="has not been loaded"):
        predictor.predict("A wonderful movie")


def test_model_without_probability_method_is_rejected(tmp_path):
    model_path = tmp_path / "incomplete.joblib"
    joblib.dump(IncompletePipeline(), model_path)
    predictor = SentimentPredictor(model_path)

    with pytest.raises(TypeError, match="predict_proba"):
        predictor.load()
    assert predictor.is_loaded is False


def test_model_failure_becomes_prediction_error(tmp_path):
    model_path = tmp_path / "broken.joblib"
    joblib.dump(BrokenPipeline(), model_path)
    predictor = SentimentPredictor(model_path)
    predictor.load()

    with pytest.raises(PredictionError, match="could not produce a prediction"):
        predictor.predict("A wonderful movie")
