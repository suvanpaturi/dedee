from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import logging
import sys
import json
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphManager():
    
    def __init__(self):
        NEO4J_URI = "bolt://128.203.120.208:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = "dedee-knowledge-graph!"
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def import_data(self, data):
        loaded_data = json.loads(data)
        with self.driver.session() as session:
            for device in loaded_data["devices"]:
                device_id = device["id"]
                device_region = device['region']
                composite_id = device_id + "-" + device_region
                session.execute_write(self.create_device, composite_id, device_id, device_region)
                for query in tqdm(device["queries"], desc=f"Adding query relationships on device {composite_id}"):
                    session.execute_write(self.create_query_relationships, composite_id, query)
    
    @staticmethod
    def create_device(tx, composite_id, device_id, device_region):
        try:
            query = (
                "MERGE (d:Device {composite_id: $composite_id}) "
                "ON CREATE SET d.device_id = $device_id, d.device_region = $device_region"
            )
            tx.run(query, composite_id=composite_id, device_id=device_id, device_region=device_region)
        except Neo4jError as e:
            logger.error(f"An error has occured when creating device {composite_id}: {str(e)}")
    
    @staticmethod
    def create_query_relationships(tx, composite_id, query):
        try:
            query_id = query["id"]
            query_text = query["query"]["text"]
            query_embedding = query["query"]["embedding"]
            answer_text = query["answer"]["text"]
            answer_embedding = query["answer"]["embedding"]
            source = query["metadata"]["source"]

            cypher_query = (
                "MATCH (d:Device {composite_id: $composite_id}) "
                "MERGE (q:Query {id: $query_id}) "
                "ON CREATE SET q.text = $query_text, q.embedding = $query_embedding "
                "MERGE (a:Answer {text: $answer_text}) "
                "ON CREATE SET a.embedding = $answer_embedding "
                "MERGE (m:Metadata {source: $source}) "
                "MERGE (d)-[:HAS_QUERY]->(q) "
                "MERGE (q)-[:HAS_ANSWER]->(a) "
                "MERGE (q)-[:HAS_METADATA]->(m)"
            )
            tx.run(cypher_query, composite_id=composite_id, query_id=query_id, query_text=query_text, query_embedding=query_embedding,
                answer_text=answer_text, answer_embedding=answer_embedding, source=source)
        except Neo4jError as e:
            logger.error(f"An error has occurred when writing query relationships on {composite_id}: {str(e)}")
            
    @DeprecationWarning    
    def add_vector_search_index(self, label, property_name, dimensions):
        with self.driver.session() as session:
          try:
              result = session.run("SHOW INDEXES")
              indexes = [record["name"] for record in result]
              
              index_name = f"{label.lower()}_{property_name}_vector_index"
              
              if index_name not in indexes:
                  session.run(
                      f"CREATE VECTOR INDEX {index_name} "
                      f"FOR (n:{label}) "
                      f"ON (n.{property_name}) "
                      f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}"
                  )
                  logger.info(f"Created vector index for {label}.{property_name}")
              else:
                  logger.info(f"Vector index for {label}.{property_name} already exists")
          except Exception as e:
              logger.warning(f"Could not create vector index: {str(e)}")