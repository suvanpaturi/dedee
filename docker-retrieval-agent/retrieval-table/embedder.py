from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Embedder(chromadb.EmbeddingFunction):
    """
    Embedding function using SentenceTransformers for ChromaDB.
    """
    
    def __init__(self, model_name):
        """
        Initialize the embedder with the specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.dimension}")
    
    def __call__(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings as numpy arrays
        """
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return [[0.0] * self.dimension] * len(texts)
    
    def get_dimension(self):
        """Get the dimension of the embedding vectors."""
        return self.dimension