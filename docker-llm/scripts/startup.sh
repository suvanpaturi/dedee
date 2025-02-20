#!/bin/bash

echo "Starting Ollama server with CPU-only mode..."
export OLLAMA_NO_ACCEL=1  # Ensure CPU usage

# Ensure the models directory exists
mkdir -p "$OLLAMA_MODELS"

# Start Ollama in the background
ollama serve &

# Wait for the server to start
sleep 5

# Read models from models.json and pull them with parameters
echo "Pulling models..."
jq -c '.models[]' /app/config/models.json | while read -r model_entry; do
    name=$(echo "$model_entry" | jq -r '.name')
    tag=$(echo "$model_entry" | jq -r '.params.tag')

    echo "Pulling model: $name:$tag ..."
    ollama pull "$name:$tag"
done

echo "Starting FastAPI server..."
uvicorn client.app:app --host 0.0.0.0 --port 8000 --workers 2 --reload
