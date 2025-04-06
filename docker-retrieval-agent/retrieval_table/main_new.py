# from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
# from pydantic import BaseModel, Field
# from neo4j import GraphDatabase
# import logging
# import numpy as np
# from typing import List, Dict, Any, Optional
# import os
# import redis
# import json
# from datetime import datetime
# import time
# import httpx
# import asyncio
# from functools import lru_cache
# from contextlib import asynccontextmanager
#
# # Import your existing components
# # from table import RetrievalTable
# from table import RetrievalTable
# from embedder import SentenceTransformerEmbedder
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# # Settings for API keys and endpoints
# class Settings:
#     # OpenAI API settings for LLM (but not for embeddings)
#     OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
#     OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
#     OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
#
#     # Embedding model (using SentenceTransformer instead of OpenAI)
#     EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
#
#     # Neo4j connection settings
#     NEO4J_URI: str = os.getenv("NEO4J_URI", "neo4j://128.203.120.208:7687")
#     NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
#     NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "dedee-knowledge-graph!")
#
#     # Retrieval settings
#     TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
#     CANDIDATE_LIMIT: int = int(os.getenv("CANDIDATE_LIMIT", "10"))
#     SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
#     HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.7"))
#     QUERY_WEIGHT: float = float(os.getenv("QUERY_WEIGHT", "0.7"))
#     ANSWER_WEIGHT: float = float(os.getenv("ANSWER_WEIGHT", "0.3"))
#
#     # ChromaDB RetrievalTable settings
#     CHROMADB_COLLECTION: str = os.getenv("CHROMADB_COLLECTION", "retrieval-table")
#     CHROMADB_MAX_ITEMS: int = int(os.getenv("CHROMADB_MAX_ITEMS", "1000"))
#     CHROMADB_PERSIST_DIR: str = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_cache")
#     CHROMADB_SIMILARITY_THRESHOLD: float = float(os.getenv("CHROMADB_SIMILARITY_THRESHOLD", "0.85"))
#
#     # Redis settings for debate mechanism
#     REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
#     REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
#     REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
#     DEBATE_QUEUE: str = os.getenv("DEBATE_QUEUE", "debate_queue")
#     JUDGEMENT_POOL: str = os.getenv("JUDGEMENT_POOL", "judgement_pool")
#     DEBATE_TIMEOUT: int = int(os.getenv("DEBATE_TIMEOUT", "300"))  # Seconds to wait for debate resolution
#
#
# @lru_cache()
# def get_settings():
#     return Settings()
#
#
# # Define lifespan to manage connections
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialize connections on startup
#     settings = get_settings()
#
#     # Initialize with SentenceTransformer model instead of OpenAI
#     app.state.kg_client = KnowledgeGraphClient(
#         settings.NEO4J_URI,
#         settings.NEO4J_USERNAME,
#         settings.NEO4J_PASSWORD,
#         openai_api_key=settings.OPENAI_API_KEY,  # Still pass this for LLM usage
#         embedding_model=settings.EMBEDDING_MODEL  # Use SentenceTransformer model
#     )
#
#     # Initialize retrieval table with ChromaDB backend
#     app.state.table = RetrievalTable(
#         collection_name=settings.CHROMADB_COLLECTION,
#         max_items=settings.CHROMADB_MAX_ITEMS,
#         persist_directory=settings.CHROMADB_PERSIST_DIR,
#         similarity_threshold=settings.CHROMADB_SIMILARITY_THRESHOLD
#     )
#
#     # Initialize Redis client for debate mechanism
#     app.state.redis_client = redis.StrictRedis(
#         host=settings.REDIS_HOST,
#         port=settings.REDIS_PORT,
#         password=settings.REDIS_PASSWORD,
#         decode_responses=True
#     )
#
#     yield
#
#     # Close connections on shutdown
#     app.state.kg_client.close()
#
#
# app = FastAPI(lifespan=lifespan)
#
#
# class KnowledgeGraphClient:
#     def __init__(self, uri, username, password, openai_api_key=None, embedding_model="all-MiniLM-L6-v2"):
#         """
#         Initialize the KnowledgeGraphClient with Neo4j connection and SentenceTransformer embeddings.
#
#         Args:
#             uri: Neo4j URI
#             username: Neo4j username
#             password: Neo4j password
#             openai_api_key: OpenAI API key for LLM (not used for embeddings)
#             embedding_model: SentenceTransformer model to use
#         """
#         self.driver = GraphDatabase.driver(uri, auth=(username, password))
#         logger.info(f"Connected to Neo4j at {uri}")
#
#         # Store the embedding model name
#         self.embedding_model = embedding_model
#
#         # Force using Python-based similarity calculations
#         self.use_neo4j_vector = False
#         logger.info("Using Python-based similarity calculations")
#
#         # Initialize SentenceTransformer for embeddings
#         try:
#             self.embedder = SentenceTransformerEmbedder(model_name=embedding_model)
#             logger.info(f"Using SentenceTransformer model: {embedding_model} with dimension {self.embedder.dimension}")
#         except Exception as e:
#             logger.error(f"Error initializing SentenceTransformerEmbedder: {str(e)}")
#             self.embedder = None
#
#         # Keep OpenAI API key for LLM usage, not for embeddings
#         self.openai_api_key = openai_api_key
#         if openai_api_key:
#             self.openai_headers = {
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {openai_api_key}"
#             }
#
#     def close(self):
#         """Close the Neo4j connection."""
#         self.driver.close()
#         logger.info("Neo4j connection closed")
#
#     def generate_embedding(self, text: str, dimensions: int = 384) -> List[float]:
#         """
#         Generate embedding for text using SentenceTransformer.
#
#         Args:
#             text: Text to generate embedding for
#             dimensions: Dimensions for fallback embedding (defaults to 384 for SentenceTransformer)
#
#         Returns:
#             Embedding vector as a list of floats
#         """
#         if self.embedder:
#             try:
#                 return self.embedder.embed(text)
#             except Exception as e:
#                 logger.error(f"Error generating SentenceTransformer embedding: {str(e)}")
#                 logger.warning("Falling back to simple embedding method")
#                 return self._generate_simple_embedding(text, dimensions)
#         else:
#             return self._generate_simple_embedding(text, dimensions)
#
#     def _generate_simple_embedding(self, text: str, dimensions: int = 384) -> List[float]:
#         """
#         Generate a simple deterministic embedding. Fallback when other methods are unavailable.
#
#         Args:
#             text: Text to generate embedding for
#             dimensions: Number of dimensions for the embedding vector (now defaults to 384)
#
#         Returns:
#             Embedding vector as a list of floats
#         """
#         # Ensure hash_value is within numpy's seed range (0 to 2^32 - 1)
#         hash_value = abs(hash(text)) % (2 ** 32 - 1)
#         np.random.seed(hash_value)
#
#         # Generate a deterministic random vector
#         embedding = np.random.randn(dimensions)
#
#         # Normalize to unit length
#         embedding = embedding / np.linalg.norm(embedding)
#
#         return embedding.tolist()
#
#     def comprehensive_search(self, query: str, settings: Settings) -> Optional[List[Dict[str, Any]]]:
#         """
#         Sequential search with exact matching first:
#         1. Check for exact text matches
#         2. Then try high-confidence semantic matches
#         3. Finally fall back to answer similarity search
#
#         Args:
#             query: User's query text
#             settings: Application settings
#
#         Returns:
#             List of dicts with query, answer and scoring info or None if no matches
#         """
#         try:
#             # STRATEGY 0: Try exact text matching first
#             with self.driver.session() as session:
#                 exact_match_result = session.run("""
#                     MATCH (q:Query)-[:HAS_ANSWER]->(a:Answer)
#                     MATCH (d:Device)-[:HAS_QUERY]->(q)
#                     WHERE toLower(q.text) = toLower($query_text)
#
#                     RETURN q.text AS query_text,
#                         a.text AS answer_text,
#                         q.id AS query_id,
#                         d.device_id AS device_id,
#                         d.device_region AS region_id,
#                         CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source
#                     LIMIT 1
#                 """, query_text=query)
#
#                 exact_matches = list(exact_match_result)
#
#                 if exact_matches:
#                     logger.info(f"Found exact text match for query: {query}")
#                     result = exact_matches[0]
#                     return [{
#                         "query_text": result["query_text"],
#                         "answer_text": result["answer_text"],
#                         "query_id": result["query_id"],
#                         "device_id": result["device_id"],
#                         "score": 1.0,  # Perfect match
#                         "query_similarity": 1.0,
#                         "answer_similarity": 0.0,
#                         "source": result["source"],
#                         "region_id": result["region_id"],
#                         "strategy": "exact_match"
#                     }]
#
#             # If no exact match, proceed with semantic matching
#             # Generate embedding for the query
#             query_embedding = self.generate_embedding(query)
#             query_embedding_np = np.array(query_embedding)
#
#             # Get all queries and answers with their embeddings
#             with self.driver.session() as session:
#                 result = session.run("""
#                     MATCH (q:Query)-[:HAS_ANSWER]->(a:Answer)
#                     MATCH (d:Device)-[:HAS_QUERY]->(q)
#                     WHERE q.embedding IS NOT NULL
#
#                     RETURN q.text AS query_text,
#                         a.text AS answer_text,
#                         q.id AS query_id,
#                         q.embedding AS query_embedding,
#                         a.embedding AS answer_embedding,
#                         d.device_id AS device_id,
#                         d.device_region AS region_id,
#                         CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source
#                 """)
#
#                 all_records = list(result)
#
#                 # For debugging
#                 logger.info(f"Retrieved {len(all_records)} records from Neo4j for semantic search")
#
#             # STRATEGY 1: High confidence direct query matches
#             high_confidence_results = []
#
#             for record in all_records:
#                 if record["query_embedding"]:
#                     q_embedding = np.array(record["query_embedding"])
#
#                     # Check if dimensions match
#                     if len(q_embedding) != len(query_embedding_np):
#                         logger.warning(
#                             f"Dimension mismatch: query embedding {len(query_embedding_np)}, stored embedding {len(q_embedding)}")
#                         continue
#
#                     q_embedding_norm = np.linalg.norm(q_embedding)
#
#                     if q_embedding_norm > 0:  # Avoid division by zero
#                         q_embedding = q_embedding / q_embedding_norm
#                         query_similarity = float(np.dot(q_embedding, query_embedding_np))
#
#                         # Debug info for high similarity
#                         if query_similarity > 0.8:
#                             logger.info(f"High similarity {query_similarity} found for query: '{record['query_text']}'")
#
#                         if query_similarity >= settings.HIGH_CONFIDENCE_THRESHOLD:
#                             high_confidence_results.append({
#                                 "query_text": record["query_text"],
#                                 "answer_text": record["answer_text"],
#                                 "query_id": record["query_id"],
#                                 "device_id": record["device_id"],
#                                 "score": query_similarity,
#                                 "query_similarity": query_similarity,
#                                 "answer_similarity": 0.0,
#                                 "source": record["source"],
#                                 "region_id": record["region_id"],
#                                 "strategy": "high_confidence"
#                             })
#
#             # If we found high confidence matches, return them immediately
#             if high_confidence_results:
#                 # Sort by score and return top-k results
#                 sorted_results = sorted(high_confidence_results, key=lambda x: x["score"], reverse=True)
#                 logger.info(f"Found {len(sorted_results)} high-confidence matches for query: {query}")
#                 return sorted_results[:settings.TOP_K_RESULTS]
#
#             # STRATEGY 2: If no high confidence matches, search answers directly
#             direct_answer_results = []
#
#             for record in all_records:
#                 if record["answer_embedding"]:
#                     a_embedding = np.array(record["answer_embedding"])
#
#                     # Check if dimensions match
#                     if len(a_embedding) != len(query_embedding_np):
#                         logger.warning(
#                             f"Dimension mismatch: answer embedding {len(a_embedding)}, query embedding {len(query_embedding_np)}")
#                         continue
#
#                     a_embedding_norm = np.linalg.norm(a_embedding)
#
#                     if a_embedding_norm > 0:
#                         a_embedding = a_embedding / a_embedding_norm
#                         answer_similarity = float(np.dot(a_embedding, query_embedding_np))
#
#                         if answer_similarity >= settings.SIMILARITY_THRESHOLD:
#                             direct_answer_results.append({
#                                 "query_text": record["query_text"],
#                                 "answer_text": record["answer_text"],
#                                 "query_id": record["query_id"],
#                                 "device_id": record["device_id"],
#                                 "score": answer_similarity,
#                                 "query_similarity": 0.0,
#                                 "answer_similarity": answer_similarity,
#                                 "source": record["source"],
#                                 "region_id": record["region_id"],
#                                 "strategy": "direct_answer"
#                             })
#
#             # If we found direct answer matches, return them
#             if direct_answer_results:
#                 # Sort by score and return top-k results
#                 sorted_results = sorted(direct_answer_results, key=lambda x: x["score"], reverse=True)
#                 logger.info(f"Found {len(sorted_results)} direct answer matches for query: {query}")
#                 return sorted_results[:settings.TOP_K_RESULTS]
#
#             # If we get here, no matches were found
#             logger.info(f"No results found for query: {query}")
#             return None
#
#         except Exception as e:
#             logger.error(f"Error in comprehensive search: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return None
#
#     def two_stage_similarity_search(self, query: str, settings: Settings) -> Optional[List[Dict[str, Any]]]:
#         """
#         Simplified method to call comprehensive_search for compatibility.
#         """
#         return self.comprehensive_search(query, settings)
#
#     def prepare_context_for_llm(self, records: List[Dict], query: str) -> str:
#         """
#         Prepare context from retrieved records for the LLM.
#
#         Args:
#             records: List of records with answers and metadata
#             query: The original user query
#
#         Returns:
#             Formatted context string
#         """
#         if not records:
#             return "No relevant information was found in the knowledge base."
#
#         context = f"The user asked: '{query}'\n\n"
#         context += "Here is relevant information from the knowledge base:\n\n"
#
#         for i, record in enumerate(records, 1):
#             context += f"[Source {i}] "
#             context += f"Question: {record['query_text']}\n"
#             context += f"Answer: {record['answer_text']}\n"
#
#             context += f"Relevance: Combined score {record['score']:.2f} "
#             context += f"(Query similarity: {record.get('query_similarity', 0):.2f}, "
#             context += f"Answer similarity: {record.get('answer_similarity', 0):.2f})\n"
#
#             if 'strategy' in record:
#                 context += f"Match strategy: {record['strategy']}\n"
#
#             # Add metadata if available
#             metadata = []
#             if record.get('source'):
#                 metadata.append(f"Source: {record['source']}")
#             if record.get('region_id'):
#                 metadata.append(f"Region: {record['region_id']}")
#
#             if metadata:
#                 context += f"Metadata: {', '.join(metadata)}\n"
#
#             context += "\n"
#
#         return context
#
#
# class LLMProcessor:
#     """Process retrieved information using an LLM to generate responses."""
#
#     @staticmethod
#     async def generate_response(query: str, context: str, settings: Settings) -> str:
#         """
#         Generate a response using an LLM based on the retrieved context.
#
#         Args:
#             query: The user's original query
#             context: Context information from the knowledge graph
#             settings: Application settings
#
#         Returns:
#             LLM-generated response
#         """
#         if not settings.OPENAI_API_KEY:
#             logger.warning("No OpenAI API key provided, returning raw context")
#
#             # Format a response from the raw context
#             if "No relevant information" in context:
#                 return "I don't have information about that in my knowledge base."
#
#             lines = context.split('\n')
#             result = "Based on the information in our knowledge base:\n\n"
#
#             # Extract just the answers from the context
#             for i, line in enumerate(lines):
#                 if line.startswith("Answer:"):
#                     result += lines[i][8:].strip() + "\n\n"
#
#             return result
#
#         try:
#             async with httpx.AsyncClient(timeout=30.0) as client:
#                 system_prompt = """
#                 You are a helpful assistant that answers questions based on the provided information.
#                 If the information doesn't fully answer the query, acknowledge the limitations.
#                 Always cite your sources when you use them, and maintain a professional tone.
#                 Do not make up information that isn't in the provided context.
#                 """
#
#                 messages = [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user",
#                      "content": f"Please answer this question: {query}\n\nHere's the information I have:\n\n{context}"}
#                 ]
#
#                 headers = {
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
#                 }
#
#                 payload = {
#                     "model": settings.OPENAI_MODEL,
#                     "messages": messages,
#                     "temperature": 0.3,
#                     "max_tokens": 500
#                 }
#
#                 response = await client.post(
#                     settings.OPENAI_API_URL,
#                     headers=headers,
#                     json=payload
#                 )
#
#                 if response.status_code != 200:
#                     logger.error(f"LLM API error: {response.text}")
#                     return f"I encountered an error processing your request. Here's what I found:\n\n{context}"
#
#                 result = response.json()
#                 return result["choices"][0]["message"]["content"]
#
#         except Exception as e:
#             logger.error(f"Error using LLM: {str(e)}")
#             return f"I encountered an error processing your request. Here's what I found:\n\n{context}"
#
#
# class DebateManager:
#     """Manage the debate process between multiple regions."""
#
#     @staticmethod
#     async def initiate_debate(query: str, region_ids: List[str], redis_client, settings: Settings) -> Optional[
#         Dict[str, Any]]:
#         """
#         Initiate a debate process between multiple regions.
#
#         Args:
#             query: The user's original query
#             region_ids: List of region IDs that should participate in the debate
#             redis_client: Redis client for queue management
#             settings: Application settings
#
#         Returns:
#             Dictionary with debate results or None if timeout
#         """
#         try:
#             # Create debate payload
#             # debate_payload = {
#             #     "query": query,
#             #     "region_ids": region_ids,
#             #     "timestamp": time.time()
#             # }
#             #
#             # # Push to debate queue
#             # redis_client.rpush(settings.DEBATE_QUEUE, json.dumps(debate_payload))
#             debate_payload = {
#                 "query": query,
#                 "region_ids": region_ids,
#                 "timestamp": time.time()
#             }
#
#             # Push to actual debate queue for parent processing
#             redis_client.rpush(settings.DEBATE_QUEUE, json.dumps(debate_payload))
#
#             # NEW: Push to metadata pool for the judge
#             redis_client.rpush("debate_meta_pool", json.dumps(debate_payload))
#
#             logger.info(f"Initiated debate for query: '{query}' with regions: {region_ids}")
#
#             # Poll for results
#             timeout = settings.DEBATE_TIMEOUT
#             start = time.time()
#
#             while time.time() - start < timeout:
#                 # Check last 20 judgments for our query
#                 judgements = redis_client.lrange(settings.JUDGEMENT_POOL, -20, -1)
#
#                 for judgement_json in reversed(judgements):
#                     try:
#                         judgement = json.loads(judgement_json)
#
#                         # Check if this judgment is for our query
#                         if judgement.get("query") and judgement["query"].lower() == query.lower():
#                             logger.info(f"Found judgment for query: '{query}'")
#                             return {
#                                 "response": judgement.get("response", "No response provided"),
#                                 "winning_region": judgement.get("winning_region"),
#                                 "confidence": judgement.get("confidence", 0.0),
#                                 "match_type": "debate_result"
#                             }
#                     except json.JSONDecodeError:
#                         # Handle string-format legacy judgments
#                         if query.lower() in judgement_json.lower():
#                             logger.info(f"Found legacy judgment for query: '{query}'")
#                             return {
#                                 "response": judgement_json,
#                                 "winning_region": None,
#                                 "confidence": 0.0,
#                                 "match_type": "debate_result_legacy"
#                             }
#
#                 # Wait before polling again
#                 await asyncio.sleep(2)
#
#             # If we get here, we timed out
#             logger.warning(f"Debate timeout for query: '{query}'")
#             return None
#
#         except Exception as e:
#             logger.error(f"Error in debate process: {str(e)}")
#             return None
#
#
# # Models for API requests and responses
# class Query(BaseModel):
#     query: str
#
#
# class QueryResponse(BaseModel):
#     query: str
#     response: str
#     sources: List[Dict[str, Any]] = Field(default_factory=list)
#     processed_by_llm: bool = False
#     match_type: Optional[str] = None
#     debate_participants: Optional[List[str]] = None
#     winning_region: Optional[str] = None
#
#
# class FeedbackRequest(BaseModel):
#     query_id: str
#     feedback: str
#     rating: int = Field(ge=1, le=5)
#
#
# # API endpoints
# @app.post("/query/", response_model=QueryResponse)
# async def handle_query(input: Query, background_tasks: BackgroundTasks, settings: Settings = Depends(get_settings)):
#     try:
#         # 1. Try to get a response from the retrieval table first (ChromaDB cache)
#         # results = app.state.table.query(query=input.query)
#         #
#         # if results:
#         #     # We found a response in the retrieval table
#         #     response = results["response"]
#         #     logger.info(
#         #         f"Retrieval table hit for query: {input.query}, match type: {results.get('match_type', 'unknown')}")
#         #
#         #     return {
#         #         "query": input.query,
#         #         "response": response,
#         #         "sources": [],
#         #         "processed_by_llm": False,
#         #         "match_type": results.get("match_type", "cache")
#         #     }
#
#         # 2. No hit in cache, use semantic search in knowledge graph
#         logger.info(f"No hit in retrieval table for query: {input.query}, performing comprehensive search")
#
#         # Get semantically similar answers using comprehensive search strategy
#         similar_results = app.state.kg_client.comprehensive_search(
#             input.query,
#             settings
#         )
#
#         if similar_results:
#             # 3. Analyze results to determine regions involved
#             region_ids = set()
#             for result in similar_results:
#                 if result.get("region_id"):
#                     region_ids.add(result["region_id"])
#
#             region_ids_list = list(region_ids)
#             logger.info(f"Query involves regions: {region_ids_list}")
#
#             # 4. Determine if multi-region debate is needed
#             if len(region_ids) > 1:
#                 logger.info(f"Multiple regions involved, initiating debate for query: {input.query}")
#
#                 # Initiate debate process
#                 debate_result = await DebateManager.initiate_debate(
#                     input.query,
#                     region_ids_list,
#                     app.state.redis_client,
#                     settings
#                 )
#
#                 if debate_result:
#                     # We got a debate result
#                     response = debate_result["response"]
#
#                     # Store debate result in cache
#                     app.state.table.put(
#                         query=input.query,
#                         response=response,
#                         # match_type=debate_result["match_type"]
#                     )
#
#                     return {
#                         "query": input.query,
#                         "response": response,
#                         "sources": [],  # No direct sources for debate results
#                         "processed_by_llm": False,  # Debate results come from regions
#                         "match_type": debate_result["match_type"],
#                         "debate_participants": region_ids_list,
#                         "winning_region": debate_result.get("winning_region")
#                     }
#                 else:
#                     # Debate timeout, return status message
#                     return {
#                         "query": input.query,
#                         "response": "Debate initiated but no judgment returned yet. Please try again later.",
#                         "sources": [],
#                         "processed_by_llm": False,
#                         "match_type": "debate_initiated",
#                         "debate_participants": region_ids_list
#                     }
#
#             # 5. Single region or no debate needed, proceed with Neo4j results
#             # Prepare context for the LLM
#             context = app.state.kg_client.prepare_context_for_llm(similar_results, input.query)
#
#             # Generate response using LLM
#             response = await LLMProcessor.generate_response(input.query, context, settings)
#
#             # Store sources for citation
#             sources = []
#             for result in similar_results:
#                 source = {
#                     "query_id": result["query_id"],
#                     "query_text": result["query_text"],
#                     "similarity": result["score"],
#                     "device_id": result["device_id"],
#                     "strategy": result.get("strategy", "unknown"),
#                     "region_id": result.get("region_id", None)
#                 }
#                 sources.append(source)
#
#             logger.info(f"Generated LLM response for query: {input.query}")
#
#             # Store the LLM-generated response in the table for future use
#             # app.state.table.put(query=input.query, response=response, match_type="neo4j_semantic")
#             app.state.table.put(query=input.query, response=response)
#
#             return {
#                 "query": input.query,
#                 "response": response,
#                 "sources": sources,
#                 "processed_by_llm": settings.OPENAI_API_KEY != "",
#                 "match_type": "neo4j_semantic",
#                 "debate_participants": region_ids_list if region_ids_list else None
#             }
#         else:
#             # 6. No information found in knowledge graph
#             logger.info(f"No information found in knowledge graph for query: {input.query}")
#
#             fallback_response = "I don't have specific information about that in my knowledge base."
#
#             # Try to use the LLM for a general response if API key is available
#             if settings.OPENAI_API_KEY:
#                 try:
#                     general_context = f"The user asked: '{input.query}', but no specific information was found in the knowledge base."
#                     fallback_response = await LLMProcessor.generate_response(
#                         input.query,
#                         general_context,
#                         settings
#                     )
#                 except Exception as e:
#                     logger.error(f"Error getting fallback response from LLM: {str(e)}")
#
#             return {
#                 "query": input.query,
#                 "response": fallback_response,
#                 "sources": [],
#                 "processed_by_llm": settings.OPENAI_API_KEY != "",
#                 "match_type": "no_match"
#             }
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return {
#             "query": input.query,
#             "response": f"An error occurred while processing your query: {str(e)}",
#             "sources": [],
#             "processed_by_llm": False,
#             "match_type": "error"
#         }
#
#
# @app.post("/put/")
# async def handle_put(input: QueryResponse):
#     app.state.table.put(query=input.query, response=input.response)
#     return {"query": input.query, "response": input.response}
#
#
# @app.get("/clear_table/")
# async def handle_clear_table():
#     app.state.table.clear_table()  # Using the correct method from your ChromaDB implementation
#     return {"message": "Table cleared"}
#
#
# @app.post("/feedback/")
# async def submit_feedback(feedback: FeedbackRequest):
#     """Submit user feedback for a query-response pair."""
#     try:
#         # Store feedback in Neo4j
#         with app.state.kg_client.driver.session() as session:
#             result = session.run("""
#                 MATCH (q:Query {id: $query_id})
#                 MERGE (f:Feedback {
#                     id: $feedback_id,
#                     text: $feedback_text,
#                     rating: $rating,
#                     timestamp: $timestamp
#                 })
#                 MERGE (q)-[r:HAS_FEEDBACK]->(f)
#                 RETURN f.id
#             """,
#                                  query_id=feedback.query_id,
#                                  feedback_id=f"feedback_{int(datetime.now().timestamp())}",
#                                  feedback_text=feedback.feedback,
#                                  rating=feedback.rating,
#                                  timestamp=datetime.now().isoformat()
#                                  )
#
#             feedback_id = result.single()[0]
#             return {"status": "success", "feedback_id": feedback_id}
#     except Exception as e:
#         logger.error(f"Error storing feedback: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")
#
#
# @app.get("/health/")
# async def health_check():
#     """Health check endpoint."""
#     try:
#         # Check Neo4j connection
#         with app.state.kg_client.driver.session() as session:
#             result = session.run("RETURN 1 as n")
#             neo4j_status = result.single() is not None
#
#         # Check LLM availability
#         llm_available = get_settings().OPENAI_API_KEY != ""
#
#         # Check ChromaDB availability
#         chroma_status = app.state.table.collection is not None
#
#         # Check Redis connection
#         redis_status = app.state.redis_client.ping()
#
#         # Get embedding model information
#         embedding_info = {
#             "model": app.state.kg_client.embedding_model,
#             "dimensions": getattr(app.state.kg_client.embedder, "dimension", "unknown")
#         }
#
#         return {
#             "status": "healthy",
#             "neo4j_connected": neo4j_status,
#             "neo4j_vector_available": app.state.kg_client.use_neo4j_vector,
#             "llm_available": llm_available,
#             "chromadb_available": chroma_status,
#             "redis_connected": redis_status,
#             "embedding": embedding_info,
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# main_new.py
from fastapi import FastAPI, HTTPException
import httpx
import asyncio
from pydantic import BaseModel

app = FastAPI()

# Configure URLs of your parents and judge here
PARENT_URLS = {
    "eastus": "http://localhost:8001/respond/",
    "westus": "http://localhost:8002/respond/",
    "westeurope": "http://localhost:8003/respond/"
}

JUDGE_URL = "http://localhost:8004/judge/"

class Query(BaseModel):
    query: str
    regions: list[str] = ["eastus", "westus", "westeurope"]  # default to all regions

@app.post("/query/")
async def handle_query(q: Query):
    selected_parents = {region: url for region, url in PARENT_URLS.items() if region in q.regions}

    if not selected_parents:
        raise HTTPException(status_code=400, detail="No valid regions specified")

    async with httpx.AsyncClient() as client:
        tasks = [client.post(url, json={"query": q.query}) for url in selected_parents.values()]
        parent_results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for result in parent_results:
        if isinstance(result, Exception) or result.status_code != 200:
            continue  # You can add error logging here
        responses.append(result.json())

    if not responses:
        raise HTTPException(status_code=500, detail="No responses received from parents")

    async with httpx.AsyncClient() as client:
        judge_res = await client.post(JUDGE_URL, json={"query": q.query, "responses": responses})

    if judge_res.status_code != 200:
        raise HTTPException(status_code=500, detail="Judge service failed")

    return judge_res.json()
