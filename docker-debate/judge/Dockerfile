FROM python:3.10-slim

WORKDIR /app

COPY judge.py .

EXPOSE 5001

RUN pip install --no-cache-dir redis requests

CMD ["python", "-u", "judge.py"]