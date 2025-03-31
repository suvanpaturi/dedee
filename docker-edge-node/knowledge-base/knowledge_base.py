import chromadb
from typing import List, Dict
from pydantic import BaseModel
from embedder import Embedder
import hashlib
from tqdm import tqdm

class Knowledge(BaseModel):
    query: str
    response: str
    source: str
        
class KnowledgeBase:
    def __init__(self, 
            collection_name: str = "knowledge-base", 
            persist_directory: str = "./chroma_cache",
            max_results: int = 10,
            embedding_function = None,
            similarity_threshold: float = 0.7):
    
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
                metadata={"description": "Edge-based Knowledge Base for queries and responses"}
            )
        
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold

    def hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()
    
    def peek(self, n):
        results = []
        peeked_results = self.collection.get(limit=n)
        for i in range(len(peeked_results["ids"])):
            results.append({
                "id": peeked_results["ids"][i],
                "query": peeked_results["documents"][i],
                "response": peeked_results["metadatas"][i]["response"],
                "source": peeked_results["metadatas"][i]["source"]
            })
        return results
        
    def get(self, query: str):
        similar_results = self.collection.query(
            query_texts=[query],
            n_results=self.max_results,
            include=["metadatas", "documents", "distances"]
        )
        
        results = []
        if similar_results["ids"]:
            for i in range(len(similar_results["ids"][0])):
                distance = similar_results["distances"][0][i]
                similarity = 1.0 - min(distance, 1.0)
                if similarity >= self.similarity_threshold:
                    results.append({
                        "id": similar_results["ids"][0][i],
                        "query": similar_results["documents"][0][i],
                        "response": similar_results["metadatas"][0][i]["response"],
                        "source": similar_results["metadatas"][0][i]["source"],
                        "similarity": similarity,
                        "match_type": "semantic"
                    })
                    
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    def update(self, knowledge_items: list[Knowledge]) -> List[Dict[str, str]]:
        all_results = []
        for i in tqdm(range(0, len(knowledge_items), 1000), desc="Adding to ChromaDB"):
            batch = knowledge_items[i:i + 1000]
            ids = [self.hash_query(item.query) for item in batch]
            self.collection.add(
                ids=ids,
                documents=[item.query for item in batch],
                metadatas=[{"response": item.response, "source": item.source} for item in batch]
            )
            results =  [
                {
                    "id": id_,
                    "query": item.query,
                    "response": item.response,
                    "source": item.source
                }
                for id_, item in zip(ids, batch)
            ]
            all_results.extend(results)
        print(f"Total queries in collection: {self.collection.count()}")
        return all_results
    
    def clear_kb(self):
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    