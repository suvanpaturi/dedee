FROM python:3.11-slim-bookworm
 
 WORKDIR /app
 
 COPY requirements.txt requirements.txt
 RUN pip install --no-cache-dir -r requirements.txt
 
 COPY . .
 
 EXPOSE 5001
 
 # Make sure uvicorn is correctly installed
 RUN pip install --no-cache-dir uvicorn
 
 # Use python to execute uvicorn instead of direct execution
 CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001", "--reload"]