import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphEmbedder():
    def __init__(self, model):
        self.model = SentenceTransformer(model)
    
    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
    def transform_edge_knowledge(self, input_data: List[Dict[str, str]], edge_id, edge_region) -> Dict[str, List[Dict[str, Any]]]:
        
        transformed_data = {
            "devices": [
                {
                    "id": edge_id,
                    "region": edge_region,
                    "queries": []
                }
            ]
        }
        
        for item in input_data:
            if not ('query' in item and 'response' in item):
                logger.warning(f"Skipping invalid item, missing query or response: {item}")
                continue
            try:
                
                query_embedding = self.get_embedding(item["query"])
                response_embedding = self.get_embedding(item["response"])
                
                query_obj = {
                    "id": item["id"],
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
                
                if "id" in item:
                    query_obj["id"] = item["id"]
                
                transformed_data["devices"][0]["queries"].append(query_obj)
                logger.info(f"Processed item: {item['query'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
        
        return transformed_data