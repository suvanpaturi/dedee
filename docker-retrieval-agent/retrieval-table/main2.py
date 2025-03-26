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
# Make sure embedder.py is available in the same directory
from table import RetrievalTable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Settings for API keys and endpoints
# Settings for API keys and endpoints
class Settings:
    # OpenAI API settings for embeddings and LLM
    OPENAI_API_KEY: str = ""
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
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
        settings.NEO4J_PASSWORD,
        openai_api_key=settings.OPENAI_API_KEY,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL
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
    def __init__(self, uri, username, password, openai_api_key=None, embedding_model="text-embedding-3-small"):
        """
        Initialize the KnowledgeGraphClient with Neo4j connection and optional OpenAI embedding capabilities.
        
        Args:
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            openai_api_key: Optional OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")
        
        # Set up embedding capabilities
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        
        # Force using Python-based similarity calculations
        self.use_neo4j_vector = False
        logger.info("Using Python-based similarity calculations")
        
        # Set up OpenAI embeddings if API key is provided
        if openai_api_key:
            import requests
            self.openai_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            }
            logger.info(f"OpenAI embeddings enabled using model {embedding_model}")
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def generate_embedding(self, text: str, dimensions: int = 1536) -> List[float]:
        """
        Generate embedding for text using OpenAI if available, fallback to simple method.
        
        Args:
            text: Text to generate embedding for
            dimensions: Dimensions for fallback embedding
            
        Returns:
            Embedding vector as a list of floats
        """
        if self.openai_api_key:
            try:
                return self._generate_openai_embedding(text)
            except Exception as e:
                logger.error(f"Error generating OpenAI embedding: {str(e)}")
                logger.warning("Falling back to simple embedding method")
                return self._generate_simple_embedding(text, dimensions)
        else:
            return self._generate_simple_embedding(text, dimensions)
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        import requests
        import time
        
        payload = {
            "model": self.embedding_model,
            "input": text,
            "encoding_format": "float"
        }
        
        # Try up to 3 times with exponential backoff
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers=self.openai_headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["data"][0]["embedding"]
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited by OpenAI API. Waiting {wait_time}s before retry.")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    raise Exception(f"OpenAI API error: {response.status_code}")
                    
            except Exception as e:
                if attempt < 2:  # Last attempt
                    wait_time = 2 ** attempt
                    logger.warning(f"Error {str(e)}. Retrying in {wait_time}s.")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception("Failed to get OpenAI embedding after multiple attempts")
    
    def _generate_simple_embedding(self, text: str, dimensions: int = 1536) -> List[float]:
        """
        Generate a simple deterministic embedding. Fallback when OpenAI is unavailable.
        Note: This will generate embeddings with the same dimensions as OpenAI for compatibility.
        
        Args:
            text: Text to generate embedding for
            dimensions: Number of dimensions for the embedding vector
            
        Returns:
            Embedding vector as a list of floats
        """
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
        Sequential search with exact matching first:
        1. Check for exact text matches
        2. Then try high-confidence semantic matches
        3. Finally fall back to answer similarity search
        
        Args:
            query: User's query text
            settings: Application settings
            
        Returns:
            List of dicts with query, answer and scoring info or None if no matches
        """
        try:
            # STRATEGY 0: Try exact text matching first
            with self.driver.session() as session:
                exact_match_result = session.run("""
                    MATCH (q:Query)-[:HAS_ANSWER]->(a:Answer)
                    MATCH (d:Device)-[:HAS_QUERY]->(q)
                    WHERE toLower(q.text) = toLower($query_text)
                    
                    RETURN q.text AS query_text, 
                        a.text AS answer_text, 
                        q.id AS query_id,
                        d.id AS device_id,
                        CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source,
                        CASE WHEN q.region IS NOT NULL THEN q.region ELSE null END AS region
                    LIMIT 1
                """, query_text=query)
                
                exact_matches = list(exact_match_result)
                
                if exact_matches:
                    logger.info(f"Found exact text match for query: {query}")
                    result = exact_matches[0]
                    return [{
                        "query_text": result["query_text"],
                        "answer_text": result["answer_text"],
                        "query_id": result["query_id"],
                        "device_id": result["device_id"],
                        "score": 1.0,  # Perfect match
                        "query_similarity": 1.0,
                        "answer_similarity": 0.0,
                        "source": result["source"],
                        "region": result["region"],
                        "strategy": "exact_match"
                    }]
            
            # If no exact match, proceed with semantic matching
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            query_embedding_np = np.array(query_embedding)
            
            # Get all queries and answers with their embeddings
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (q:Query)-[:HAS_ANSWER]->(a:Answer)
                    MATCH (d:Device)-[:HAS_QUERY]->(q)
                    WHERE q.embedding IS NOT NULL
                    
                    RETURN q.text AS query_text, 
                        a.text AS answer_text, 
                        q.id AS query_id,
                        q.embedding AS query_embedding,
                        a.embedding AS answer_embedding,
                        d.id AS device_id,
                        CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source,
                        CASE WHEN q.region IS NOT NULL THEN q.region ELSE null END AS region
                """)
                
                all_records = list(result)
                
                # For debugging
                logger.info(f"Retrieved {len(all_records)} records from Neo4j for semantic search")
            
            # STRATEGY 1: High confidence direct query matches
            high_confidence_results = []
            
            for record in all_records:
                if record["query_embedding"]:
                    q_embedding = np.array(record["query_embedding"])
                    
                    # Check if dimensions match
                    if len(q_embedding) != len(query_embedding_np):
                        logger.warning(f"Dimension mismatch: query embedding {len(query_embedding_np)}, stored embedding {len(q_embedding)}")
                        continue
                    
                    q_embedding_norm = np.linalg.norm(q_embedding)
                    
                    if q_embedding_norm > 0:  # Avoid division by zero
                        q_embedding = q_embedding / q_embedding_norm
                        query_similarity = float(np.dot(q_embedding, query_embedding_np))
                        
                        # Debug info for high similarity
                        if query_similarity > 0.8:
                            logger.info(f"High similarity {query_similarity} found for query: '{record['query_text']}'")
                        
                        if query_similarity >= settings.HIGH_CONFIDENCE_THRESHOLD:
                            high_confidence_results.append({
                                "query_text": record["query_text"],
                                "answer_text": record["answer_text"],
                                "query_id": record["query_id"],
                                "device_id": record["device_id"],
                                "score": query_similarity,
                                "query_similarity": query_similarity,
                                "answer_similarity": 0.0,
                                "source": record["source"],
                                "region": record["region"],
                                "strategy": "high_confidence"
                            })
            
            # If we found high confidence matches, return them immediately
            if high_confidence_results:
                # Sort by score and return top-k results
                sorted_results = sorted(high_confidence_results, key=lambda x: x["score"], reverse=True)
                logger.info(f"Found {len(sorted_results)} high-confidence matches for query: {query}")
                return sorted_results[:settings.TOP_K_RESULTS]
            
            # STRATEGY 2: If no high confidence matches, search answers directly
            direct_answer_results = []
            
            for record in all_records:
                if record["answer_embedding"]:
                    a_embedding = np.array(record["answer_embedding"])
                    
                    # Check if dimensions match
                    if len(a_embedding) != len(query_embedding_np):
                        continue
                    
                    a_embedding_norm = np.linalg.norm(a_embedding)
                    
                    if a_embedding_norm > 0:
                        a_embedding = a_embedding / a_embedding_norm
                        answer_similarity = float(np.dot(a_embedding, query_embedding_np))
                        
                        if answer_similarity >= settings.SIMILARITY_THRESHOLD:
                            direct_answer_results.append({
                                "query_text": record["query_text"],
                                "answer_text": record["answer_text"],
                                "query_id": record["query_id"],
                                "device_id": record["device_id"],
                                "score": answer_similarity,
                                "query_similarity": 0.0,
                                "answer_similarity": answer_similarity,
                                "source": record["source"],
                                "region": record["region"],
                                "strategy": "direct_answer"
                            })
            
            # If we found direct answer matches, return them
            if direct_answer_results:
                # Sort by score and return top-k results
                sorted_results = sorted(direct_answer_results, key=lambda x: x["score"], reverse=True)
                logger.info(f"Found {len(sorted_results)} direct answer matches for query: {query}")
                return sorted_results[:settings.TOP_K_RESULTS]
            
            # If we get here, no matches were found
            logger.info(f"No results found for query: {query}")
            return None
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {str(e)}")
            import traceback
            traceback.print_exc()  # Add this for more detailed error info
            return None
    
    def two_stage_similarity_search(self, query: str, settings: Settings) -> Optional[List[Dict[str, Any]]]:
        """
        Simplified method to call comprehensive_search for compatibility.
        """
        return self.comprehensive_search(query, settings)
    
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
            context += f"[Source {i}] "
            context += f"Question: {record['query_text']}\n"
            context += f"Answer: {record['answer_text']}\n"
            
            context += f"Relevance: Combined score {record['score']:.2f} "
            context += f"(Query similarity: {record.get('query_similarity', 0):.2f}, "
            context += f"Answer similarity: {record.get('answer_similarity', 0):.2f})\n"
            
            if 'strategy' in record:
                context += f"Match strategy: {record['strategy']}\n"
            
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

# Models for API requests and responses
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

# API endpoints
@app.post("/query/", response_model=QueryResponse)
async def handle_query(input: Query, background_tasks: BackgroundTasks, settings: Settings = Depends(get_settings)):
    # Try to get a response from the retrieval table first using ChromaDB backend
    results = app.state.table.query(query=input.query)
    
    if results:
        # We found a response in the retrieval table
        response = results["response"]
        logger.info(f"Retrieval table hit for query: {input.query}, match type: {results.get('match_type', 'unknown')}")
        
        return {
            "query": input.query, 
            "response": response,
            "sources": [],
            "processed_by_llm": False,
            "match_type": results.get("match_type", None)
        }
    else:
        # No hit in the retrieval table, use semantic search in knowledge graph
        logger.info(f"No hit in retrieval table for query: {input.query}, performing comprehensive search")
        
        # Get semantically similar answers using comprehensive search strategy
        similar_results = app.state.kg_client.comprehensive_search(
            input.query, 
            settings
        )
        
        if similar_results:
            # Prepare context for the LLM
            context = app.state.kg_client.prepare_context_for_llm(similar_results, input.query)
            
            # Generate response using LLM
            response = await LLMProcessor.generate_response(input.query, context, settings)
            
            # Store sources for citation
            sources = []
            for result in similar_results:
                source = {
                    "query_id": result["query_id"],
                    "query_text": result["query_text"],
                    "similarity": result["score"],
                    "device_id": result["device_id"],
                    "strategy": result.get("strategy", "unknown")
                }
                sources.append(source)
            
            logger.info(f"Generated LLM response for query: {input.query}")
            
            # Store the LLM-generated response in the table for future use
            app.state.table.put(query=input.query, response=response)
            
            return {
                "query": input.query,
                "response": response,
                "sources": sources,
                "processed_by_llm": settings.OPENAI_API_KEY != "",
                "match_type": "neo4j_semantic"
            }
        else:
            logger.info(f"No information found in knowledge graph for query: {input.query}")
            
            fallback_response = "I don't have specific information about that in my knowledge base."
            
            # Try to use the LLM for a general response if API key is available
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