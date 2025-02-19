FROM ubuntu:22.04

# Install dependencies with fewer layers and better cleanup
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies with CPU-specific packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U huggingface_hub

# Create model directories
RUN mkdir -p /app/models/tinyllama /app/models/sentence-transformer

# Download models with explicit CPU configuration
RUN python3 -c "from huggingface_hub import snapshot_download; \
    tinyllama_path = snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
                                      local_dir='/app/models/tinyllama', \
                                      local_dir_use_symlinks=False); \
    st_path = snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', \
                              local_dir='/app/models/sentence-transformer', \
                              local_dir_use_symlinks=False); \
    print(f'Models downloaded successfully to {tinyllama_path} and {st_path}')"

# Verify models were downloaded correctly
RUN ls -la /app/models/tinyllama/config.json && \
    ls -la /app/models/sentence-transformer/config.json

# Copy application code
COPY . .

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_db

# Set environment variables for CPU operation
ENV PYTORCH_DEVICE=cpu
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV PYTHONUNBUFFERED=1
ENV TINYLLAMA_PATH=/app/models/tinyllama
ENV ST_PATH=/app/models/sentence-transformer
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]