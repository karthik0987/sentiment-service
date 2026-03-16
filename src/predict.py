import joblib

model = joblib.load("models/model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")


def predict_sentiment(text: str) -> dict:
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    return {
        "text": text,
        "sentiment": "positive" if prediction == 1 else "negative",
        "confidence": round(float(max(probability)), 4),
    }
