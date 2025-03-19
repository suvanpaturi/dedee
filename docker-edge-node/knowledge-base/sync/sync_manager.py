from knowledge_base import Knowledge
from embedder import Embedder
from neo4j import GraphDatabase
import os
from concurrent.futures import ThreadPoolExecutor

EDGE_REGION = os.getenv("INSTANCE_REGION", "unknown-region")
EDGE_ID = os.getenv("INSTANCE_ID", "unknown-instance")

class GraphSyncManager():
    def __init__(self):
        NEO4J_URI = "bolt://128.203.120.208:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = "dedee-global-knowledge-graph!"
        self.embedding_function = Embedder("all-MiniLM-L6-v2")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def insert_knowledge(self, data: Knowledge):
        encode_question = self.embedding_function([data.query])
        encode_answer = self.embedding_function([data.response])
        
        with self.driver.session() as session:
            session.run(
            """
            MERGE (q:Question {text: $question, embedding: $question_embedding})
            MERGE (a:Answer {text: $answer, embedding: $answer_embedding})
            MERGE (m:Metadata {document_source: $source, device_source: $device, region: $region})
            MERGE (q)-[:ANSWERED_BY]->(a)
            MERGE (q)-[:HAS_Metadata]->(m)
            """,
            question=data.query,
            question_embedding=encode_question,
            answer=data.response,
            answer_embedding=encode_answer,
            source=data.source,
            device=EDGE_ID,
            region=EDGE_REGION
            )
        session.close()
        print(f"Successfully inserted knowledge from edge device {EDGE_ID} to knowledge-graph")

    def insert_parallel(self, knowledge_data, num_workers=5):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(self.insert_knowledge, knowledge_data)
