# from chromadb.api.types import Documents, Embeddings
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer
#
# class Embedder(embedding_functions.EmbeddingFunction):
#     def __init__(self, model_name: str):
#         self.model = SentenceTransformer(model_name)
#
#     def __call__(self, input: Documents) -> Embeddings:
#         return self.model.encode(input).tolist()


from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import logging
from typing import List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Embedder(chromadb.EmbeddingFunction):
    """
    Embedding function using SentenceTransformers for ChromaDB.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
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


class SentenceTransformerEmbedder:
    """
    Class to generate embeddings using the SentenceTransformer library.
    This class provides a similar interface to the Embedder class but with
    additional functionality for single text embedding.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the SentenceTransformer embedder.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        # Reuse the Embedder class to avoid code duplication
        self.embedder = Embedder(model_name)
        self.model = self.embedder.model
        self.dimension = self.embedder.dimension
        self.model_name = model_name

    def embed(self, text: Union[str, List[str]], batch_size: int = 32) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text or a list of texts.

        Args:
            text: Text or list of texts to embed
            batch_size: Batch size for processing multiple texts

        Returns:
            Embedding vector(s) as a list of floats or list of lists of floats
        """
        # Handle different input types
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text

        # Use the embedder to generate embeddings
        embeddings = self.embedder(texts)

        # Return single embedding if input was a single text
        if is_single_text:
            return embeddings[0]
        else:
            return embeddings
