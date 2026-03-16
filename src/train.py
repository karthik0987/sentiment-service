import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(n_samples=10000):
    """Pull data from HuggingFace — not a static CSV."""
    print("Loading IMDB dataset from HuggingFace...")
    dataset = load_dataset("imdb", split="train")
    df = pd.DataFrame(dataset).sample(n=n_samples, random_state=42)
    df.to_csv("data/imdb_sample.csv", index=False)
    print(f"Loaded {len(df)} samples")
    return df


def train():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run(run_name="tfidf-logreg-v1"):
        max_features = 5000
        C = 1.0

        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        mlflow.log_param("n_samples", len(df))

        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        print("Training model...")
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    print("Model and vectorizer saved to /models")


if __name__ == "__main__":
    train()
