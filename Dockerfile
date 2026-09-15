FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    jinja2==3.1.4 \
    python-multipart==0.0.9 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    shap==0.45.1 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    reportlab==4.2.0

RUN mkdir -p /app/web/static

EXPOSE 8000
CMD uvicorn web.main:app --host 0.0.0.0 --port $PORT
