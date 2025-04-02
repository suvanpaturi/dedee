FROM python:3.10-slim

WORKDIR /app

COPY debate.py .

COPY judge.py .

RUN pip install --no-cache-dir redis requests

CMD ["python", "-u", "debate.py"]
