#!/bin/bash

# Convenience script to process and upload edge device data to Neo4j
# Usage: ./run.sh input.json

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Error: Please provide an input JSON file"
  echo "Usage: ./run.sh input.json [device_id]"
  exit 1
fi

INPUT_FILE=$1

# Device ID defaults to edge-device-1 or can be specified as second argument
DEVICE_ID="${2:-edge-device-1}"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file $INPUT_FILE not found"
  exit 1
fi

# Create temp file for the processed data
TEMP_FILE="$(mktemp -t neo4j_data.XXXXXX.json)"

echo "1. Generating embeddings for $INPUT_FILE..."
python embedding/embedding_generator.py --input "$INPUT_FILE" --output "$TEMP_FILE" --device "$DEVICE_ID"

if [ $? -ne 0 ]; then
  echo "Error: Failed to generate embeddings"
  rm -f "$TEMP_FILE"
  exit 1
fi

echo "2. Uploading data to Neo4j..."
python neo4j/neo4j_uploader.py --file "$TEMP_FILE"

if [ $? -ne 0 ]; then
  echo "Error: Failed to upload data to Neo4j"
  rm -f "$TEMP_FILE"
  exit 1
fi

echo "3. Cleaning up temporary files..."
rm -f "$TEMP_FILE"

echo "Process completed successfully!"