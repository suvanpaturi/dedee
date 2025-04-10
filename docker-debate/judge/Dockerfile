FROM python:3.10-slim

WORKDIR /app

COPY judge.py .

EXPOSE 5001

RUN pip install --no-cache-dir fastapi requests uvicorn

CMD ["python", "-m", "uvicorn", "judge:app", "--host", "0.0.0.0", "--port", "5001"]