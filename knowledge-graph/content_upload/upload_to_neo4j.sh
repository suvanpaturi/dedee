#!/bin/bash

# Script to upload content to Neo4j
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
NEO4J_URI="neo4j://128.203.120.208:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="dedee-knowledge-graph!"
CONTENT_FILE="knowledge_content.json"
GENERATE_EMBEDDINGS=false
EMBEDDING_DIMENSIONS=8

# Function to print status message
print_status() {
  echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to print warning message
print_warning() {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Function to print success message
print_success() {
  echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to print error message
print_error() {
  echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --uri)
      NEO4J_URI="$2"
      shift
      shift
      ;;
    --username)
      NEO4J_USERNAME="$2"
      shift
      shift
      ;;
    --password)
      NEO4J_PASSWORD="$2"
      shift
      shift
      ;;
    --file)
      CONTENT_FILE="$2"
      shift
      shift
      ;;
    --embed)
      GENERATE_EMBEDDINGS=true
      shift
      ;;
    --dimensions)
      EMBEDDING_DIMENSIONS="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --uri URI             Neo4j URI (default: neo4j://128.203.120.208:7687)"
      echo "  --username USER       Neo4j username (default: neo4j)"
      echo "  --password PASS       Neo4j password"
      echo "  --file FILE           JSON content file (default: knowledge_content.json)"
      echo "  --embed               Generate embeddings for queries and answers using Neo4j"
      echo "  --dimensions DIM      Dimensions for embeddings (default: 8)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if the content file exists
if [ ! -f "$CONTENT_FILE" ]; then
  print_error "Content file not found: $CONTENT_FILE"
  exit 1
fi

# Install required Python packages if needed
print_status "Checking required Python packages..."
if ! pip show neo4j > /dev/null 2>&1; then
  print_status "Installing neo4j Python package..."
  pip install neo4j
fi

if ! pip show numpy > /dev/null 2>&1; then
  print_status "Installing numpy Python package..."
  pip install numpy
fi

if ! pip show requests > /dev/null 2>&1; then
  print_status "Installing requests Python package..."
  pip install requests
fi

# Check for neo4j-vector plugin
print_status "Checking for Neo4j-Vector plugin..."
NEO4J_VECTOR_CHECK=$(python -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USERNAME', '$NEO4J_PASSWORD'))
    with driver.session() as session:
        result = session.run(\"CALL dbms.procedures() YIELD name WHERE name CONTAINS 'neo4j.vector' OR name CONTAINS 'graphvector' RETURN count(*) > 0 as has_vector\")
        record = result.single()
        print('true' if record and record['has_vector'] else 'false')
    driver.close()
except Exception as e:
    print('error: ' + str(e))
")

# Generate embeddings if requested
if [ "$GENERATE_EMBEDDINGS" = true ]; then
  if [[ "$NEO4J_VECTOR_CHECK" == *"error"* ]]; then
    print_error "Failed to check for Neo4j-Vector plugin: ${NEO4J_VECTOR_CHECK#error: }"
    print_status "Will attempt to generate embeddings anyway..."
  elif [ "$NEO4J_VECTOR_CHECK" = "false" ]; then
    print_warning "Neo4j-Vector plugin not detected. Will use fallback embedding method."
  else
    print_success "Neo4j-Vector plugin detected and will be used for embeddings."
  fi

  print_status "Generating embeddings for queries and answers..."
  
  # Create a temporary file for the output
  TEMP_FILE="${CONTENT_FILE%.json}_with_embeddings.json"
  
  python embedding_generator.py \
    --input "$CONTENT_FILE" \
    --output "$TEMP_FILE" \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-username "$NEO4J_USERNAME" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --dimensions "$EMBEDDING_DIMENSIONS"
  
  if [ $? -eq 0 ]; then
    print_success "Embeddings successfully generated"
    CONTENT_FILE="$TEMP_FILE"
  else
    print_error "Failed to generate embeddings"
    exit 1
  fi
fi

print_status "Neo4j URI: $NEO4J_URI"

# Run the Python uploader script
print_status "Uploading content to Neo4j..."
python neo4j_content_upload.py \
  --uri "$NEO4J_URI" \
  --username "$NEO4J_USERNAME" \
  --password "$NEO4J_PASSWORD" \
  --file "$CONTENT_FILE"

# Check if upload was successful
if [ $? -eq 0 ]; then
  print_success "Content successfully uploaded to Neo4j!"
else
  print_error "Failed to upload content to Neo4j"
  exit 1
fi

print_success "Upload process completed!"

# Generate embeddings if requested
if [ "$GENERATE_EMBEDDINGS" = true ]; then
  if [[ "$NEO4J_VECTOR_CHECK" == *"error"* ]]; then
    print_error "Failed to check for Neo4j-Vector plugin: ${NEO4J_VECTOR_CHECK#error: }"
    print_status "Will attempt to generate embeddings anyway..."
  elif [ "$NEO4J_VECTOR_CHECK" = "false" ]; then
    print_warning "Neo4j-Vector plugin not detected. Will use fallback embedding method."
  else
    print_success "Neo4j-Vector plugin detected and will be used for embeddings."
  fi

  print_status "Generating embeddings for queries and answers..."
  
  # Create a temporary file for the output
  TEMP_FILE="${CONTENT_FILE%.json}_with_embeddings.json"
  
  python embedding_generator.py \
    --input "$CONTENT_FILE" \
    --output "$TEMP_FILE" \
    --neo4j-uri "$NEO4J_URI" \
    --neo4j-username "$NEO4J_USERNAME" \
    --neo4j-password "$NEO4J_PASSWORD" \
    --dimensions "$EMBEDDING_DIMENSIONS"
  
  if [ $? -eq 0 ]; then
    print_success "Embeddings successfully generated"
    CONTENT_FILE="$TEMP_FILE"
  else
    print_error "Failed to generate embeddings"
    exit 1
  fi
fi

print_status "Neo4j URI: $NEO4J_URI"

# Run the Python uploader script
print_status "Uploading content to Neo4j..."
python neo4j_content_upload.py \
  --uri "$NEO4J_URI" \
  --username "$NEO4J_USERNAME" \
  --password "$NEO4J_PASSWORD" \
  --file "$CONTENT_FILE"

# Check if upload was successful
if [ $? -eq 0 ]; then
  print_success "Content successfully uploaded to Neo4j!"
else
  print_error "Failed to upload content to Neo4j"
  exit 1
fi

print_success "Upload process completed!"