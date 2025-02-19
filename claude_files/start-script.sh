#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
sleep 5

# Pull the specified model
ollama pull $MODEL

# Keep container running
tail -f /dev/null
