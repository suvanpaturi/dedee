#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama service..."
sleep 5

# Run model management script
python3 /manage_models.py

# Monitor Ollama process
while true; do
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Ollama process died, restarting..."
        ollama serve &
        sleep 5
        python3 /manage_models.py
    fi
    sleep 10
done
