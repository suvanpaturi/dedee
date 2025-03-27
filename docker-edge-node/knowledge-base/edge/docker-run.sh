#!/bin/bash

# Remove any existing container
docker rm -f knowledge-base 2>/dev/null

# Build the Docker image
docker build -t knowledge-base .

# Run the container with sleep infinity to keep it alive
docker run -d \
  -p 5001:5001 \
  -e OPENAI_API_KEY="sk-proj-IYXxFiUbWs0CYZ31caQlT3BlbkFJQp03evB1IVRb1zbDHKoj" \
  -e NEO4J_URI="neo4j://128.203.120.208:7687" \
  -e NEO4J_USERNAME="neo4j" \
  -e NEO4J_PASSWORD="dedee-knowledge-graph!" \
  -e DEVICE_ID="docker-edge-device-1" \
  --name knowledge-base \
  knowledge-base
  
# Wait for container to start
sleep 2

# Run embedding generation and upload directly
echo "Generating embeddings and uploading to Neo4j..."
docker exec knowledge-base python embedding_generator.py --input sample_data.json --output temp_data.json
docker exec knowledge-base python neo4j_uploader.py --file temp_data.json

echo "Process completed"