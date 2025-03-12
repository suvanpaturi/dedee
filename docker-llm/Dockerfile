# Set the base image to Ubuntu
FROM ubuntu:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_MODELS=/app/models
ENV OLLAMA_NO_ACCEL=1

# Install dependencies
RUN apt update && apt install -y curl wget jq python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | bash

# Create the models directory
RUN mkdir -p $OLLAMA_MODELS

# Create a virtual environment for Python
RUN python3 -m venv /app/venv

# Set the virtual environment path
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies inside the virtual environment
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Set working directory
WORKDIR /app

# Copy project files
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY client/ /app/client/

# Ensure the startup script is executable
RUN chmod +x /app/scripts/startup.sh

# Expose API Ports
EXPOSE 11434 8000

# Start Ollama and FastAPI using the virtual environment
CMD ["/bin/bash", "/app/scripts/startup.sh"]
