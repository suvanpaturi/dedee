FROM python:3.10-slim

WORKDIR /app

COPY parent.py .

EXPOSE 5001

RUN pip install --no-cache-dir fastapi requests uvicorn pydantic httpx

CMD ["python", "-m", "uvicorn", "parent:app", "--host", "0.0.0.0", "--port", "5001"]