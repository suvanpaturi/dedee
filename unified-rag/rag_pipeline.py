from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os
import requests

app = FastAPI(title="Unified RAG Pipeline")

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-retrieval-agent-eastus:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dedee-knowledge-graph!")
NODE_ENDPOINTS = {
    "retrieval-agent-eastus": "http://retrieval-agent-eastus:8000/query/",
    "retrieval-agent-westeurope": "http://retrieval-agent-westeurope:8000/query/",
    "retrieval-agent-westus": "http://retrieval-agent-westus:8000/query/",
    "docker-llm": "http://docker-llm-client:5001/api/generate"
}

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

class Query(BaseModel):
    query: str
    top_k: int = 5

class RagResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

@app.post("/rag/query", response_model=RagResponse)
async def unified_query(query: Query):
    """
    Unified RAG pipeline that queries across all nodes
    """
    query_text = query.query
    query_embedding = model.encode(query_text).tolist()
    
    # Phase 1: Search Neo4j for existing answers
    with neo4j_driver.session() as session:
        # First try vector similarity search
        result = session.run("""
            MATCH (k:Knowledge)
            WHERE k.embedding IS NOT NULL
            WITH k, gds.similarity.cosine(k.embedding, $embedding) AS score
            WHERE score > 0.7
            RETURN k.query AS query, k.response AS response, 
                   k.source AS source, k.node AS node, score
            ORDER BY score DESC
            LIMIT $top_k
        """, embedding=query_embedding, top_k=query.top_k)
        
        knowledge_items = []
        for record in result:
            knowledge_items.append({
                "query": record["query"],
                "response": record["response"],
                "source": record["source"],
                "node": record["node"],
                "score": record["score"]
            })
    
    # Phase 2: If no results from Neo4j, query individual nodes
    if not knowledge_items:
        for node_name, endpoint in NODE_ENDPOINTS.items():
            if "llm" in node_name.lower():
                # Skip LLM nodes for direct querying
                continue
                
            try:
                response = requests.post(
                    endpoint, 
                    json={"query": query_text},
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("response"):
                        knowledge_items.append({
                            "query": query_text,
                            "response": result["response"],
                            "source": node_name,
                            "score": result.get("similarity", 0.8)
                        })
                        
                        # Store this in Neo4j for future use
                        with neo4j_driver.session() as session:
                            session.run("""
                                MERGE (k:Knowledge {query: $query, node: $node})
                                SET k.response = $response,
                                    k.embedding = $embedding,
                                    k.source = $node,
                                    k.updated_at = datetime()
                            """, query=query_text, response=result["response"], 
                                 embedding=query_embedding, node=node_name)
            except Exception as e:
                print(f"Error querying {node_name}: {str(e)}")
    
    # Phase 3: Generate the final answer using LLM
    if knowledge_items:
        # Prepare context from retrieved items
        context = "\n\n".join([
            f"Source ({item['source']}): {item['response']}"
            for item in knowledge_items
        ])
        
        # Find LLM endpoint
        llm_endpoint = next((endpoint for node, endpoint in NODE_ENDPOINTS.items() 
                           if "llm" in node.lower()), None)
        
        if llm_endpoint:
            try:
                llm_response = requests.post(
                    llm_endpoint,
                    json={
                        "prompt": f"Based on the following information, answer the question: '{query_text}'\n\nContext:\n{context}\n\nAnswer:",
                        "max_tokens": 500
                    },
                    timeout=10
                )
                
                if llm_response.status_code == 200:
                    answer = llm_response.json().get("response", "")
                    return {
                        "answer": answer,
                        "sources": knowledge_items
                    }
            except Exception as e:
                print(f"Error generating answer with LLM: {str(e)}")
        
        # Fallback if LLM fails
        return {
            "answer": knowledge_items[0]["response"],
            "sources": knowledge_items
        }
    
    return {
        "answer": "I don't have enough information to answer this question.",
        "sources": []
    }

@app.post("/rag/add")
async def add_knowledge(knowledge: Dict[str, Any]):
    """
    Add knowledge to the central graph
    """
    query = knowledge.get("query")
    response = knowledge.get("response")
    source = knowledge.get("source", "manual")
    
    if not query or not response:
        raise HTTPException(status_code=400, detail="Query and response are required")
    
    try:
        # Generate embedding
        embedding = model.encode(query).tolist()
        
        # Store in Neo4j
        with neo4j_driver.session() as session:
            session.run("""
                MERGE (k:Knowledge {query: $query})
                SET k.response = $response,
                    k.embedding = $embedding,
                    k.source = $source,
                    k.node = $source,
                    k.updated_at = datetime()
            """, query=query, response=response, embedding=embedding, 
                 source=source, node=source)
        
        return {"status": "success", "message": "Knowledge added to graph"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding knowledge: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    neo4j_driver.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)