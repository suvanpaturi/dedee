from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import os
import requests
import json
from datetime import datetime
import httpx
import asyncio
from functools import lru_cache
from contextlib import asynccontextmanager

# Import your existing RetrievalTable
from table import RetrievalTable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Settings for API keys and endpoints
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Neo4j connection settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "neo4j://128.203.120.208:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "dedee-knowledge-graph!")
    
    # Retrieval settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    CANDIDATE_LIMIT: int = int(os.getenv("CANDIDATE_LIMIT", "10")) 
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
    QUERY_WEIGHT: float = float(os.getenv("QUERY_WEIGHT", "0.7"))  
    ANSWER_WEIGHT: float = float(os.getenv("ANSWER_WEIGHT", "0.3"))
    
    # ChromaDB RetrievalTable settings
    CHROMADB_COLLECTION: str = os.getenv("CHROMADB_COLLECTION", "retrieval-table")
    CHROMADB_MAX_ITEMS: int = int(os.getenv("CHROMADB_MAX_ITEMS", "1000"))
    CHROMADB_PERSIST_DIR: str = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_cache")
    CHROMADB_SIMILARITY_THRESHOLD: float = float(os.getenv("CHROMADB_SIMILARITY_THRESHOLD", "0.85"))

@lru_cache()
def get_settings():
    return Settings()

# Define lifespan to manage connections
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize connections on startup
    settings = get_settings()
    app.state.kg_client = KnowledgeGraphClient(
        settings.NEO4J_URI, 
        settings.NEO4J_USERNAME, 
        settings.NEO4J_PASSWORD
    )
    
    # Initialize retrieval table with ChromaDB backend
    app.state.table = RetrievalTable(
        collection_name=settings.CHROMADB_COLLECTION,
        max_items=settings.CHROMADB_MAX_ITEMS,
        persist_directory=settings.CHROMADB_PERSIST_DIR,
        similarity_threshold=settings.CHROMADB_SIMILARITY_THRESHOLD
    )
    
    yield
    
    # Close connections on shutdown
    app.state.kg_client.close()

app = FastAPI(lifespan=lifespan)

class KnowledgeGraphClient:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")
        
        # Check if Neo4j Vector is available and configure embedding method
        self.use_neo4j_vector = self._check_vector_capabilities()
        if self.use_neo4j_vector:
            logger.info("Using Neo4j-Vector for embeddings")
        else:
            logger.info("Using fallback embedding method")
    
    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def _check_vector_capabilities(self) -> bool:
        """Check if Neo4j has vector capabilities."""
        try:
            with self.driver.session() as session:
                # Try to detect Neo4j Vector procedures
                result = session.run(
                    "CALL dbms.procedures() "
                    "YIELD name "
                    "WHERE name CONTAINS 'neo4j.vector' OR name CONTAINS 'graphvector' "
                    "RETURN count(*) > 0 as has_vector"
                )
                record = result.single()
                return record and record["has_vector"]
        except Exception as e:
            logger.warning(f"Error checking Neo4j Vector capabilities: {str(e)}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        if self.use_neo4j_vector:
            return self._generate_neo4j_embedding(text)
        else:
            return self._generate_simple_embedding(text)
    
    def _generate_neo4j_embedding(self, text: str) -> List[float]:
        """Generate embedding using Neo4j Vector."""
        try:
            with self.driver.session() as session:
                # Try the neo4j.vector procedure
                result = session.run(
                    "CALL neo4j.vector.generate($text) "
                    "YIELD vector "
                    "RETURN vector AS embedding",
                    text=text
                )
                record = result.single()
                if record and "embedding" in record:
                    return record["embedding"]
                
                # Try the graphvector procedure as fallback
                result = session.run(
                    "CALL graphvector.generate.embedding($text) "
                    "YIELD embedding "
                    "RETURN embedding",
                    text=text
                )
                record = result.single()
                if record and "embedding" in record:
                    return record["embedding"]
                
                # If no embedding was generated, use the simple method
                return self._generate_simple_embedding(text)
        except Exception as e:
            logger.error(f"Error generating Neo4j embedding: {str(e)}")
            return self._generate_simple_embedding(text)
    
    def _generate_simple_embedding(self, text: str, dimensions: int = 8) -> List[float]:
        """Generate a simple deterministic embedding."""
        # Ensure hash_value is within numpy's seed range (0 to 2^32 - 1)
        hash_value = abs(hash(text)) % (2**32 - 1)
        np.random.seed(hash_value)
        
        # Generate a deterministic random vector
        embedding = np.random.randn(dimensions)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def comprehensive_search(self, query: str, settings: Settings) -> Optional[List[Dict[str, Any]]]:
        """
        Comprehensive search combining multiple strategies.
        
        Args:
            query: User's query text
            settings: Application settings
            
        Returns:
            List of dicts with query, answer and scoring info or None if no matches
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Collect results from all search methods
            all_results = []
            
            with self.driver.session() as session:
                # STRATEGY 1: Direct answer similarity (search through all answers)
                direct_answer_result = session.run("""
                    MATCH (q:Query)-[:HAS_ANSWER]->(a:Answer)
                    MATCH (d:Device)-[:HAS_QUERY]->(q)
                    WHERE a.embedding IS NOT NULL
                    WITH a, q, d, neo4j.vector.similarity(a.embedding, $query_embedding) AS answer_similarity
                    WHERE answer_similarity >= $threshold
                    
                    RETURN q.text AS query_text, 
                           a.text AS answer_text, 
                           q.id AS query_id,
                           answer_similarity AS score,
                           0.0 AS query_similarity,
                           answer_similarity,
                           d.id AS device_id,
                           CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source,
                           CASE WHEN q.region IS NOT NULL THEN q.region ELSE null END AS region,
                           "direct_answer" AS strategy
                """, 
                query_embedding=query_embedding,
                threshold=settings.SIMILARITY_THRESHOLD
                )
                
                all_results.extend([dict(record) for record in direct_answer_result])
                
                # STRATEGY 2: Two-stage query+answer similarity
                two_stage_result = session.run("""
                    MATCH (q:Query)
                    WHERE q.embedding IS NOT NULL
                    WITH q, neo4j.vector.similarity(q.embedding, $query_embedding) AS query_similarity
                    WHERE query_similarity >= $threshold
                    
                    MATCH (q)-[:HAS_ANSWER]->(a:Answer)
                    MATCH (d:Device)-[:HAS_QUERY]->(q)
                    
                    WITH q, a, d, query_similarity,
                         CASE WHEN a.embedding IS NOT NULL 
                              THEN neo4j.vector.similarity(a.embedding, $query_embedding) 
                              ELSE 0 END AS answer_similarity
                    
                    WITH q, a, d, query_similarity, answer_similarity,
                         (query_similarity * $query_weight) + (answer_similarity * $answer_weight) AS combined_score
                    
                    RETURN q.text AS query_text, 
                           a.text AS answer_text, 
                           q.id AS query_id,
                           combined_score AS score,
                           query_similarity,
                           answer_similarity,
                           d.id AS device_id,
                           CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source,
                           CASE WHEN q.region IS NOT NULL THEN q.region ELSE null END AS region,
                           "two_stage" AS strategy
                """, 
                query_embedding=query_embedding,
                threshold=settings.SIMILARITY_THRESHOLD,
                query_weight=settings.QUERY_WEIGHT,
                answer_weight=settings.ANSWER_WEIGHT
                )
                
                all_results.extend([dict(record) for record in two_stage_result])
                
                # STRATEGY 3: High-confidence direct query matches
                high_confidence_result = session.run("""
                    MATCH (q:Query)
                    WHERE q.embedding IS NOT NULL
                    WITH q, neo4j.vector.similarity(q.embedding, $query_embedding) AS similarity
                    WHERE similarity >= $high_threshold
                    
                    MATCH (q)-[:HAS_ANSWER]->(a:Answer)
                    MATCH (d:Device)-[:HAS_QUERY]->(q)
                    
                    RETURN q.text AS query_text, 
                           a.text AS answer_text, 
                           q.id AS query_id,
                           similarity AS score,
                           similarity AS query_similarity,
                           0.0 AS answer_similarity,
                           d.id AS device_id,
                           CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source,
                           CASE WHEN q.region IS NOT NULL THEN q.region ELSE null END AS region,
                           "high_confidence" AS strategy
                """, 
                query_embedding=query_embedding,
                high_threshold=settings.HIGH_CONFIDENCE_THRESHOLD
                )
                
                all_results.extend([dict(record) for record in high_confidence_result])
            
            # Remove duplicates (same query_id)
            seen_ids = set()
            unique_results = []
            
            for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
                if result["query_id"] not in seen_ids:
                    seen_ids.add(result["query_id"])
                    unique_results.append(result)
            
            if not unique_results:
                logger.info(f"No results found for query: {query}")
                return None
            
            # Return top-k results
            return unique_results[:settings.TOP_K_RESULTS]
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {str(e)}")
            return None
    
    def prepare_context_for_llm(self, records: List[Dict], query: str) -> str:
        """
        Prepare context from retrieved records for the LLM.
        
        Args:
            records: List of records with answers and metadata
            query: The original user query
            
        Returns:
            Formatted context string
        """
        if not records:
            return "No relevant information was found in the knowledge base."
        
        context = f"The user asked: '{query}'\n\n"
        context += "Here is relevant information from the knowledge base:\n\n"
        
        for i, record in enumerate(records, 1):
            context += f"[Source {i} - {record.get('strategy', 'unknown')}] "
            context += f"Question: {record['query_text']}\n"
            context += f"Answer: {record['answer_text']}\n"
            
            context += f"Relevance: Combined score {record['score']:.2f} "
            context += f"(Query similarity: {record.get('query_similarity', 0):.2f}, "
            context += f"Answer similarity: {record.get('answer_similarity', 0):.2f})\n"
            
            # Add metadata if available
            metadata = []
            if record.get('source'):
                metadata.append(f"Source: {record['source']}")
            if record.get('region'):
                metadata.append(f"Region: {record['region']}")
                
            if metadata:
                context += f"Metadata: {', '.join(metadata)}\n"
            
            context += "\n"
        
        return context

class LLMProcessor:
    """Process retrieved information using an LLM to generate responses."""
    
    @staticmethod
    async def generate_response(query: str, context: str, settings: Settings) -> str:
        """
        Generate a response using an LLM based on the retrieved context.
        
        Args:
            query: The user's original query
            context: Context information from the knowledge graph
            settings: Application settings
            
        Returns:
            LLM-generated response
        """
        if not settings.OPENAI_API_KEY:
            logger.warning("No OpenAI API key provided, returning raw context")
            
            # Format a response from the raw context
            if "No relevant information" in context:
                return "I don't have information about that in my knowledge base."
                
            lines = context.split('\n')
            result = "Based on the information in our knowledge base:\n\n"
            
            # Extract just the answers from the context
            for i, line in enumerate(lines):
                if line.startswith("Answer:"):
                    result += lines[i][8:].strip() + "\n\n"
            
            return result

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                system_prompt = """
                You are a helpful assistant that answers questions based on the provided information. 
                If the information doesn't fully answer the query, acknowledge the limitations.
                Always cite your sources when you use them, and maintain a professional tone.
                Do not make up information that isn't in the provided context.
                """
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please answer this question: {query}\n\nHere's the information I have:\n\n{context}"}
                ]
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
                }
                
                payload = {
                    "model": settings.OPENAI_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 500
                }
                
                response = await client.post(
                    settings.OPENAI_API_URL,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"LLM API error: {response.text}")
                    return f"I encountered an error processing your request. Here's what I found:\n\n{context}"
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Error using LLM: {str(e)}")
            return f"I encountered an error processing your request. Here's what I found:\n\n{context}"
        
class Query(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    processed_by_llm: bool = False
    match_type: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str
    rating: int = Field(ge=1, le=5)

# API Endpoints
@app.post("/query/", response_model=QueryResponse)
async def handle_query(input: Query, background_tasks: BackgroundTasks, settings: Settings = Depends(get_settings)):
    # Step 1: Check Retrieval Table (ChromaDB)
    results = app.state.table.query(query=input.query)
    
    if results:
        # Found in retrieval table
        return {
            "query": input.query, 
            "response": results["response"],
            "sources": [],
            "processed_by_llm": False,
            "match_type": results.get("match_type", None)
        }
    
    # Step 2: Perform Comprehensive Search in Knowledge Graph
    similar_results = app.state.kg_client.comprehensive_search(
        input.query, 
        settings
    )
    
    if similar_results:
        # Prepare context for LLM
        context = app.state.kg_client.prepare_context_for_llm(similar_results, input.query)
        
        # Generate response using LLM
        response = await LLMProcessor.generate_response(input.query, context, settings)
        
        # Prepare sources
        sources = [{
            "query_id": result["query_id"],
            "query_text": result["query_text"],
            "similarity": result["score"],
            "device_id": result["device_id"],
            "strategy": result.get("strategy", "unknown")
        } for result in similar_results]
        
        # Store in retrieval table
        app.state.table.put(query=input.query, response=response)
        
        return {
            "query": input.query,
            "response": response,
            "sources": sources,
            "processed_by_llm": settings.OPENAI_API_KEY != "",
            "match_type": "comprehensive_search"
        }
    
    # Fallback: Generate generic response
    fallback_response = "I couldn't find specific information about your query in the knowledge base."
    
    if settings.OPENAI_API_KEY:
        try:
            general_context = f"The user asked: '{input.query}', but no specific information was found in the knowledge base."
            fallback_response = await LLMProcessor.generate_response(
                input.query, 
                general_context,
                settings
            )
        except Exception as e:
            logger.error(f"Error getting fallback response from LLM: {str(e)}")
    
    return {
        "query": input.query,
        "response": fallback_response,
        "sources": [],
        "processed_by_llm": settings.OPENAI_API_KEY != "",
        "match_type": "no_match"
    }

@app.post("/put/")
async def handle_put(input: QueryResponse):
    app.state.table.put(query=input.query, response=input.response)
    return {"query": input.query, "response": input.response}

@app.get("/clear_table/")
async def handle_clear_table():
    app.state.table.clear_table()  # Using the correct method from your ChromaDB implementation
    return {"message": "Table cleared"}

@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for a query-response pair."""
    try:
        # Store feedback in Neo4j
        with app.state.kg_client.driver.session() as session:
            result = session.run("""
                MATCH (q:Query {id: $query_id})
                MERGE (f:Feedback {
                    id: $feedback_id,
                    text: $feedback_text,
                    rating: $rating,
                    timestamp: $timestamp
                })
                MERGE (q)-[r:HAS_FEEDBACK]->(f)
                RETURN f.id
            """,
            query_id=feedback.query_id,
            feedback_id=f"feedback_{int(datetime.now().timestamp())}",
            feedback_text=feedback.feedback,
            rating=feedback.rating,
            timestamp=datetime.now().isoformat()
            )
            
            feedback_id = result.single()[0]
            return {"status": "success", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Neo4j connection
        with app.state.kg_client.driver.session() as session:
            result = session.run("RETURN 1 as n")
            neo4j_status = result.single() is not None
        
        # Check LLM availability
        llm_available = get_settings().OPENAI_API_KEY != ""
        
        # Check ChromaDB availability
        chroma_status = app.state.table.collection is not None
        
        return {
            "status": "healthy",
            "neo4j_connected": neo4j_status,
            "neo4j_vector_available": app.state.kg_client.use_neo4j_vector,
            "llm_available": llm_available,
            "chromadb_available": chroma_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Optional: Main block for running the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)