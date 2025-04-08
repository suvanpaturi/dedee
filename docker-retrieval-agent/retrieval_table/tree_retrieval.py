from neo4j import GraphDatabase
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import os
from embedder import SentenceTransformerEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Settings for API keys and endpoints
class RetrievalSettings:

    # Neo4j connection settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "neo4j://128.203.120.208:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "dedee-knowledge-graph!")

    # Retrieval settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    CANDIDATE_LIMIT: int = int(os.getenv("CANDIDATE_LIMIT", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.7"))
    QUERY_WEIGHT: float = float(os.getenv("QUERY_WEIGHT", "0.7"))
    ANSWER_WEIGHT: float = float(os.getenv("ANSWER_WEIGHT", "0.3"))

class TreeRetriever:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        settings = RetrievalSettings
        
        self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD))
        logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")

        self.embedding_model = embedding_model

        self.use_neo4j_vector = False
        logger.info("Using Python-based similarity calculations")

        try:
            self.embedder = SentenceTransformerEmbedder(model_name=embedding_model)
            logger.info(f"Using SentenceTransformer model: {embedding_model} with dimension {self.embedder.dimension}")
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformerEmbedder: {str(e)}")
            self.embedder = None

    def generate_embedding(self, text: str, dimensions: int = 384) -> List[float]:
        if self.embedder:
            try:
                return self.embedder.embed(text)
            except Exception as e:
                logger.error(f"Error generating SentenceTransformer embedding: {str(e)}")
                logger.warning("Falling back to simple embedding method")
                return self._generate_simple_embedding(text, dimensions)
        else:
            return self.generate_simple_embedding(text, dimensions)

    def generate_simple_embedding(self, text: str, dimensions: int = 384) -> List[float]:
        # Ensure hash_value is within numpy's seed range (0 to 2^32 - 1)
        hash_value = abs(hash(text)) % (2 ** 32 - 1)
        np.random.seed(hash_value)

        # Generate a deterministic random vector
        embedding = np.random.randn(dimensions)

        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    def comprehensive_search(self, query: str) -> Optional[List[Dict[str, Any]]]:
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
        settings = RetrievalSettings
        
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
                        d.device_id AS device_id,
                        d.device_region AS region_id,
                        CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source
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
                        "region_id": result["region_id"],
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
                        d.device_id AS device_id,
                        d.device_region AS region_id,
                        CASE WHEN q.source IS NOT NULL THEN q.source ELSE null END AS source
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
                        logger.warning(
                            f"Dimension mismatch: query embedding {len(query_embedding_np)}, stored embedding {len(q_embedding)}")
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
                                "region_id": record["region_id"],
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
                        logger.warning(
                            f"Dimension mismatch: answer embedding {len(a_embedding)}, query embedding {len(query_embedding_np)}")
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
                                "region_id": record["region_id"],
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
            traceback.print_exc()
            return None