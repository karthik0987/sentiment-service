FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python src/train.py

EXPOSE 8000 8501

CMD uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

Now open `.gitignore` and paste:
```
venv/
__pycache__/
mlruns/
mlartifacts/
*.pyc
data/
models/
```

And update `requirements.txt` with the actual packages:
```
scikit-learn
pandas
datasets
mlflow
fastapi
uvicorn
streamlit
requests
joblib