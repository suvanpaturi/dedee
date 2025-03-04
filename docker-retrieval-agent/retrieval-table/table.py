import chromadb
from embedder import Embedder
import time
import hashlib

class RetrievalTable:
    def __init__(self, 
            collection_name: str = "retrieval-table", 
            max_items: int = 1000, 
            persist_directory: str = "./chroma_cache", 
            embedding_function = None,
            similarity_threshold: float = 0.85):
    
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
                metadata={"description": "LRU Cache-based Retrieval Table for queries and responses"}
            )
        
        self.max_items = max_items
        self.semilarity_threshold = similarity_threshold

    def hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()
    
    def query(self, query: str) -> str:
        
        query_id = self.hash_query(query)
        
        results = self.collection.get(
            ids=[query_id],
            include=["metadatas", "documents"]
        )
        
        if results["ids"]: #Return exact match
            timestamp = time.time()
            self.collection.update(
                ids=[query_id],
                metadatas=[{"timestamp": timestamp, **results["metadatas"][0]}]
            )
            
            return {
                "query": results["documents"][0],
                "response": results["metadatas"][0]["response"],
                "cached_at": results["metadatas"][0]["original_timestamp"],
                "match_type": "exact"
            }
        
        similar_results = self.collection.query(
            query_texts=[query],
            n_results=1,
            include=["metadatas", "documents", "distances"]
        )
            
        if similar_results["ids"] and similar_results["ids"][0]:
            distance = similar_results["distances"][0][0]
            similarity = 1.0 - min(distance, 1.0)
                
            if similarity >= self.similarity_threshold:
                similar_id = similar_results["ids"][0][0]
                    
                timestamp = time.time()
                self.collection.update(
                    ids=[similar_id],
                    metadatas=[{"timestamp": timestamp, **similar_results["metadatas"][0][0]}]
                )
                    
                response = {
                    "query": similar_results["documents"][0][0],
                    "response": similar_results["metadatas"][0][0]["response"],
                    "cached_at": similar_results["metadatas"][0][0]["original_timestamp"],
                    "similarity": similarity,
                    "match_type": "semantic"
                }
                return response
        
        return None
    
    def put(self, query: str, response: str):
        query_id = self.hash_query(query)
        timestamp = time.time()
        self.collection.add(
            ids=[query_id],
            metadatas=[{"response": response, "timestamp": timestamp}],
            documents=[query]
        )

        if self.collection.count() > self.max_items:
            self.evict_table()
        try:
            self.collection.update(
                ids=[query_id],
                documents=[query],
                metadatas=[{
                    "response": response,
                    "timestamp": timestamp,
                    "original_timestamp": timestamp
                }]
            )
        except:
            self.collection.add(
                ids=[query_id],
                documents=[query],
                metadatas=[{
                    "response": response,
                    "timestamp": timestamp,
                    "original_timestamp": timestamp
                }]
            )

    def evict_table(self):

        evict_count = evict_count = max(1, int(self.max_items * 0.1))

        results = self.collection.get(
                limit=self.collection.count(),
                include=["metadatas", "documents"]
        )
        sorted_items = sorted(
                zip(results["ids"], results["metadatas"]), 
                key=lambda x: x[1].get("timestamp", 0)
            )
        evict_items = [item[0] for item in sorted_items[:evict_count]]

        if evict_items:
                self.collection.delete(ids=evict_items)

    def clear_table(self):
        self.collection.delete_all()