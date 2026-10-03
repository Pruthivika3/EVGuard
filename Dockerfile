FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    matplotlib \
    seaborn \
    shap \
    joblib \
    optuna \
    openpyxl

COPY app.py .
COPY models ./models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]