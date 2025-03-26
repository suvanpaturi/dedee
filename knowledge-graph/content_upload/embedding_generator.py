import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional
import requests
from neo4j import GraphDatabase

# Configure logging
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
    
    def process_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Process a JSON file and add embeddings.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (if None, will modify the input file)
            
        Returns:
            The processed data with embeddings
        """
        try:
            # Read the input file
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Check if it has the expected format
            if 'devices' not in data:
                logger.error("Input file does not have the expected format with 'devices' key")
                return None
            
            # Collect all queries and answers that need embeddings
            query_texts = []
            answer_texts = []
            device_queries = []
            
            for device in data['devices']:
                for query_data in device['queries']:
                    # Check if query needs embedding
                    if 'query' in query_data and 'text' in query_data['query']:
                        query_text = query_data['query']['text']
                        if 'embedding' not in query_data['query'] or not query_data['query']['embedding']:
                            query_texts.append(query_text)
                            device_queries.append((device['id'], query_data))
                    
                    # Check if answer needs embedding
                    if 'answer' in query_data and 'text' in query_data['answer']:
                        answer_text = query_data['answer']['text']
                        if 'embedding' not in query_data['answer'] or not query_data['answer']['embedding']:
                            answer_texts.append(answer_text)
            
            # Get embeddings for queries
            if query_texts:
                logger.info(f"Generating embeddings for {len(query_texts)} queries")
                query_embeddings = self.batch_process_embeddings(query_texts)
                
                # Update query data with embeddings
                for i, (device_id, query_data) in enumerate(device_queries):
                    if i < len(query_embeddings):
                        query_data['query']['embedding'] = query_embeddings[i]
                    
                    # Generate ID if not present
                    if 'id' not in query_data:
                        import hashlib
                        hash_object = hashlib.md5(query_data['query']['text'].encode())
                        query_data['id'] = f"{device_id}_{hash_object.hexdigest()[:8]}"
            
            # Get embeddings for answers
            if answer_texts:
                logger.info(f"Generating embeddings for {len(answer_texts)} answers")
                answer_embeddings = self.batch_process_embeddings(answer_texts)
                
                # Update answer data with embeddings
                i = 0
                for device in data['devices']:
                    for query_data in device['queries']:
                        if 'answer' in query_data and 'text' in query_data['answer']:
                            if 'embedding' not in query_data['answer'] or not query_data['answer']['embedding']:
                                if i < len(answer_embeddings):
                                    query_data['answer']['embedding'] = answer_embeddings[i]
                                    i += 1
            
            # Write the updated data to the output file
            output_path = output_file or input_file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Generated {len(query_texts)} query embeddings and {len(answer_texts)} answer embeddings")
            logger.info(f"Saved updated data to {output_path}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

def main():

    api_key = ""

    parser = argparse.ArgumentParser(description="Generate OpenAI embeddings for query-answer pairs")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", help="Output JSON file (defaults to overwriting input)")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model name")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for API requests")
    
    args = parser.parse_args()
    
    
    generator = OpenAIEmbeddingGenerator(
        api_key=api_key,
        model=args.model
    )
    
    try:
        generator.process_file(args.input, args.output)
        logger.info("Embedding generation complete")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    main()