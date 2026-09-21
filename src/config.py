import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str
    model_path: Path
    max_review_length: int


def load_settings() -> Settings:
    configured_model_path = os.getenv("MODEL_PATH")
    model_path = (
        Path(configured_model_path).expanduser().resolve()
        if configured_model_path
        else PROJECT_ROOT / "models" / "pipeline.joblib"
    )

    return Settings(
        app_name=os.getenv("APP_NAME", "Sentiment Analysis API"),
        model_path=model_path,
        max_review_length=int(os.getenv("MAX_REVIEW_LENGTH", "10000")),
    )


settings = load_settings()
