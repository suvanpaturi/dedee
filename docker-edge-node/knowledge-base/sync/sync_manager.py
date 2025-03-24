from knowledge_base import Knowledge
from embedder import Embedder
from neo4j import GraphDatabase
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

EDGE_ID = os.getenv("EDGE_ID", "default-id")
EDGE_REGION = os.getenv("EDGE_REGION", "default-region")


class GraphSyncManager():
    def __init__(self):
        NEO4J_URI = "bolt://128.203.120.208:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = "dedee-knowledge-graph!"
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
        print(f"Successfully inserted knowledge from edge device {EDGE_ID} in region {EDGE_REGION} to knowledge-graph")

    async def insert_parallel(self, knowledge_data, num_workers=5):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            await asyncio.gather(*[
                loop.run_in_executor(executor, self.insert_knowledge, k)
                for k in knowledge_data
            ])
