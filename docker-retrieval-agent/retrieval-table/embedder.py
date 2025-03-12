from chromadb.api.types import Documents, Embeddings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class Embedder(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()