from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import chromadb
from embedder import Embedder
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Knowledge(BaseModel):
    """Knowledge item with query, response, and optional embeddings."""
    query: str
    response: str
    query_embedding: Optional[List[float]] = None
    response_embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeBase:
    """Local knowledge base using ChromaDB."""
    
    def __init__(self, 
                collection_name: str = "knowledge-base", 
                persist_directory: str = "./chroma_cache",
                embedding_function = None):
        """
        Initialize the knowledge base.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_function: Function to use for embeddings
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        if embedding_function is None:
            self.embedding_function = Embedder("all-MiniLM-L6-v2")
        else:
            self.embedding_function = embedding_function

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Local knowledge base for edge device"}
            )
        
        logger.info(f"Initialized knowledge base with collection '{collection_name}'")
    
    def update(self, items: List[Knowledge]) -> List[Knowledge]:
        """
        Update the knowledge base with new knowledge items.
        
        Args:
            items: List of Knowledge items to add
            
        Returns:
            List of Knowledge items with embeddings added
        """
        # Store embeddings and query -> response mappings
        ids = []
        documents = []
        metadatas = []
        added_items = []
        
        for item in items:
            # Generate embeddings if not provided
            if item.query_embedding is None:
                item.query_embedding = self.embedding_function([item.query])[0]
            
            if item.response_embedding is None:
                item.response_embedding = self.embedding_function([item.response])[0]
            
            # Create a unique ID based on the query
            item_id = self._hash_query(item.query)
            
            ids.append(item_id)
            documents.append(item.query)
            
            # Prepare metadata with response and embeddings
            metadata = {
                "response": item.response,
                "query_embedding": item.query_embedding,
                "response_embedding": item.response_embedding
            }
            
            # Add any additional metadata
            if item.metadata:
                metadata.update(item.metadata)
            
            metadatas.append(metadata)
            added_items.append(item)
        
        # Update the collection
        if ids:
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Added {len(ids)} items to knowledge base")
            except Exception as e:
                logger.error(f"Error adding items to knowledge base: {str(e)}")
        
        return added_items
    
    def get(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.
        
        Args:
            query: Query string
            n_results: Number of results to return
            
        Returns:
            List of results with query, response, and similarity score
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # Calculate similarity score (1.0 - distance)
                distance = results["distances"][0][i]
                similarity = 1.0 - min(distance, 1.0)
                
                result = {
                    "query": results["documents"][0][i],
                    "response": results["metadatas"][0][i]["response"],
                    "similarity": similarity
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def clear_kb(self):
        """Clear all items from the knowledge base."""
        self.collection.delete()
        logger.info("Knowledge base cleared")
    
    def _hash_query(self, query: str) -> str:
        """Create a hash of the query for a unique ID."""
        return hashlib.sha256(query.encode()).hexdigest()