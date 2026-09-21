import os
import time

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def load_data(n_samples=10000):
    """Pull data from HuggingFace — not a static CSV."""
    print("Loading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb", split="train")
    df = pd.DataFrame(dataset).sample(n=n_samples, random_state=42)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/imdb_sample.csv", index=False)
    print(f"Loaded {len(df)} samples")
    return df


def validate_data(df):
    """Verify that the training data has the expected structure."""
    required_columns = {"text", "label"}
    actual_columns = set(df.columns)

    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("The dataset is empty")

    if df["text"].isna().any():
        raise ValueError("The dataset contains missing review text")

    if df["text"].str.strip().eq("").any():
        raise ValueError("The dataset contains empty reviews")

    if df["label"].isna().any():
        raise ValueError("The dataset contains missing labels")

    expected_labels = {0, 1}
    actual_labels = set(df["label"].unique())

    if actual_labels != expected_labels:
        raise ValueError(
            f"Expected labels {expected_labels}, but found {actual_labels}"
        )

    label_counts = df["label"].value_counts()
    if label_counts.min() < 2:
        raise ValueError("Each label needs at least two rows for a stratified split")

    duplicate_count = df["text"].duplicated().sum()

    print("\nDataset validation passed")
    print(f"Rows: {len(df)}")
    print(f"Duplicate reviews: {duplicate_count}")
    print("Label distribution:")
    print(df["label"].value_counts())


def build_pipeline(classifier, max_features=5000):
    """Combine TF-IDF preprocessing and a classifier."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                stop_words="english",
            ),
        ),
        ("classifier", classifier),
    ])


def train():
    df = load_data()
    validate_data(df)

    original_row_count = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    removed_duplicates = original_row_count - len(df)
    print(f"Removed duplicate reviews: {removed_duplicates}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print("Training label proportions:")
    print(y_train.value_counts(normalize=True))
    print("Test label proportions:")
    print(y_test.value_counts(normalize=True))

    max_features = 5000
    candidate_models = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000),
        "multinomial_naive_bayes": MultinomialNB(alpha=1.0),
        "calibrated_linear_svm": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0),
            method="sigmoid",
            cv=3,
            n_jobs=-1,
        ),
        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_rows = []

    mlflow.set_experiment("sentiment-analysis-model-comparison")

    for model_name, classifier in candidate_models.items():
        print(f"\nEvaluating {model_name}...")
        pipeline = build_pipeline(classifier, max_features=max_features)
        started_at = time.perf_counter()
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            n_jobs=-1,
            error_score="raise",
        )
        duration_seconds = time.perf_counter() - started_at

        result = {
            "model": model_name,
            "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
            "cv_accuracy_std": cv_results["test_accuracy"].std(),
            "cv_f1_macro_mean": cv_results["test_f1_macro"].mean(),
            "cv_f1_macro_std": cv_results["test_f1_macro"].std(),
            "duration_seconds": duration_seconds,
        }
        comparison_rows.append(result)

        with mlflow.start_run(run_name=f"comparison-{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_metric("cv_accuracy_mean", result["cv_accuracy_mean"])
            mlflow.log_metric("cv_accuracy_std", result["cv_accuracy_std"])
            mlflow.log_metric("cv_f1_macro_mean", result["cv_f1_macro_mean"])
            mlflow.log_metric("cv_f1_macro_std", result["cv_f1_macro_std"])
            mlflow.log_metric("duration_seconds", duration_seconds)

        print(f"Accuracy: {result['cv_accuracy_mean']:.4f} +/- {result['cv_accuracy_std']:.4f}")
        print(f"Macro F1: {result['cv_f1_macro_mean']:.4f} +/- {result['cv_f1_macro_std']:.4f}")
        print(f"Comparison time: {duration_seconds:.2f} seconds")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by="cv_f1_macro_mean",
        ascending=False,
    )
    print("\nModel comparison (ranked by mean CV macro F1):")
    print(comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["model"]
    best_classifier = candidate_models[best_model_name]
    best_pipeline = build_pipeline(best_classifier, max_features=max_features)

    print(f"\nTraining selected model: {best_model_name}")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    probabilities = best_pipeline.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    predictions = pd.DataFrame({
        "text": X_test.to_numpy(),
        "true_label": y_test.to_numpy(),
        "predicted_label": y_pred,
        "confidence": confidence,
    })
    errors = predictions[
        predictions["true_label"] != predictions["predicted_label"]
    ].copy()
    errors["error_type"] = errors.apply(
        lambda row: "false_positive"
        if row["predicted_label"] == 1
        else "false_negative",
        axis=1,
    )
    errors = errors.sort_values(by="confidence", ascending=False)

    os.makedirs("reports", exist_ok=True)
    error_report_path = "reports/error_analysis.csv"
    errors.to_csv(error_report_path, index=False)

    matrix = confusion_matrix(y_test, y_pred)
    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])

    with mlflow.start_run(run_name=f"selected-{best_model_name}"):
        mlflow.log_param("model", best_model_name)
        mlflow.log_param("selection_metric", "cv_f1_macro_mean")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", report["weighted avg"]["precision"])
        mlflow.log_metric("test_recall", report["weighted avg"]["recall"])
        mlflow.log_metric("test_f1_macro", report["macro avg"]["f1-score"])
        mlflow.log_metric("false_positives", false_positives)
        mlflow.log_metric("false_negatives", false_negatives)
        mlflow.log_artifact(error_report_path, artifact_path="error_analysis")
        mlflow.sklearn.log_model(
            best_pipeline,
            name="sentiment_pipeline",
        )

    print(f"Selected model: {best_model_name}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(matrix)
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Error report saved to {error_report_path}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/pipeline.joblib")
    print("Selected pipeline saved to models/pipeline.joblib")


if __name__ == "__main__":
    train()
