import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIEmbeddingGenerator:
    """Generate embeddings for text using OpenAI's API."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding generator with OpenAI.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Test the API connection
        try:
            self._get_embedding("Test connection")
            logger.info(f"Successfully connected to OpenAI API using model {model}")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {str(e)}")
            raise
        
    def _get_embedding(self, text: str, retry_count: int = 3, backoff_factor: float = 2.0) -> List[float]:
        """
        Get embedding from OpenAI with retry logic.
        
        Args:
            text: Text to generate embedding for
            retry_count: Number of retries on failure
            backoff_factor: Exponential backoff factor
            
        Returns:
            Embedding vector as a list of floats
        """
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        
        for attempt in range(retry_count + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["data"][0]["embedding"]
                elif response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limited by OpenAI API. Attempt {attempt + 1}/{retry_count + 1}")
                    if attempt < retry_count:
                        wait_time = backoff_factor ** attempt
                        logger.info(f"Waiting {wait_time:.2f} seconds before retry")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Rate limit exceeded after {retry_count} retries")
                else:
                    error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                if attempt < retry_count:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Error {str(e)}. Retrying in {wait_time:.2f} seconds. Attempt {attempt + 1}/{retry_count + 1}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to get embedding after {retry_count} retries: {str(e)}")
                    raise
        
        raise Exception(f"Failed to get embedding after {retry_count} retries")
    
    def batch_process_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Process a batch of texts to get embeddings.
        
        Args:
            texts: List of texts to get embeddings for
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for text in batch:
                try:
                    embedding = self._get_embedding(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error getting embedding for text: {text[:50]}...: {str(e)}")
                    # Provide a fallback empty embedding
                    batch_embeddings.append([])
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return embeddings
    
    def transform_edge_data(self, input_data: List[Dict[str, Any]], device_id: str = "edge-device-1") -> Dict[str, Any]:
        """
        Transform edge device data format to Neo4j uploader format and add embeddings.
        
        Args:
            input_data: List of items from edge device
            device_id: ID of the edge device
            
        Returns:
            Transformed data in Neo4j format with embeddings
        """
        # Prepare transformed data structure
        transformed_data = {
            "devices": [
                {
                    "id": device_id,
                    "queries": []
                }
            ]
        }
        
        # Process each item
        for item in input_data:
            # Skip invalid items
            if not ('query' in item and 'response' in item):
                logger.warning(f"Skipping invalid item, missing query or response: {item}")
                continue
                
            try:
                # Generate embeddings
                query_embedding = self._get_embedding(item["query"])
                response_embedding = self._get_embedding(item["response"])
                
                # Create query object in expected format
                query_obj = {
                    "query": {
                        "text": item["query"],
                        "embedding": query_embedding
                    },
                    "answer": {
                        "text": item["response"],
                        "embedding": response_embedding
                    },
                    "metadata": {
                        "source": item.get("source", "unknown"),
                    }
                }
                
                # Add ID if available
                if "id" in item:
                    query_obj["id"] = item["id"]
                
                # Add to devices array
                transformed_data["devices"][0]["queries"].append(query_obj)
                logger.info(f"Processed item: {item['query'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
        
        return transformed_data
    
    def process_file(self, input_file: str, output_file: str = None, device_id: str = None) -> Dict[str, Any]:
        """
        Process a JSON file and add embeddings.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (if None, will modify the input file)
            device_id: ID of the edge device (if None, will use default)
            
        Returns:
            The processed data with embeddings
        """
        try:
            # Read the input file
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                # Simple list format from edge device
                device_id = device_id or "edge-device-1"
                transformed_data = self.transform_edge_data(data, device_id)
                
            elif isinstance(data, dict) and 'devices' in data:
                # Already in Neo4j format, just add embeddings
                transformed_data = data
                
                # Collect all queries and answers that need embeddings
                for device in transformed_data['devices']:
                    for query_data in device['queries']:
                        # Check if query needs embedding
                        if 'query' in query_data and 'text' in query_data['query']:
                            query_text = query_data['query']['text']
                            if 'embedding' not in query_data['query'] or not query_data['query']['embedding']:
                                query_data['query']['embedding'] = self._get_embedding(query_text)
                        
                        # Check if answer needs embedding
                        if 'answer' in query_data and 'text' in query_data['answer']:
                            answer_text = query_data['answer']['text']
                            if 'embedding' not in query_data['answer'] or not query_data['answer']['embedding']:
                                query_data['answer']['embedding'] = self._get_embedding(answer_text)
            else:
                logger.error("Input file format not recognized")
                return None
            
            # Write the updated data to the output file
            output_path = output_file or input_file
            with open(output_path, 'w') as f:
                json.dump(transformed_data, f, indent=2)
            
            logger.info(f"Processed data and saved to {output_path}")
            
            return transformed_data
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

def main():
    # Get API key from environment variable or .env file
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    # Load from .env file if not in environment
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("OPENAI_API_KEY", "")
        except ImportError:
            pass
    
    # Check if API key is available
    if not api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or create a .env file.")
        return

    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings for query-answer pairs")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file (defaults to overwriting input)")
    parser.add_argument("--device", help="Device ID to use")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model name")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for API requests")
    
    args = parser.parse_args()
    
    generator = OpenAIEmbeddingGenerator(
        api_key=api_key,
        model=args.model
    )
    
    try:
        generator.process_file(args.input, args.output, args.device)
        logger.info("Embedding generation complete")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    main()