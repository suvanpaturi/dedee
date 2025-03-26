FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_pipeline.py .

EXPOSE 8080

CMD ["uvicorn", "rag_pipeline:app", "--host", "0.0.0.0", "--port", "8080"]