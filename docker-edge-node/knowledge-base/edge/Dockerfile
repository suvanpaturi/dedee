FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY edge_device/edge_upload.py .
COPY embedding/embedding_generator.py .
COPY neo4j/neo4j_uploader.py .
COPY sample_data.json .

# Environment variables will be provided at runtime

# Expose port 5001 for API
EXPOSE 5001

# Command to run the application
CMD ["sleep", "infinity"]