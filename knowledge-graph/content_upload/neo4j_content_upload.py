import argparse
import json
import logging
import os
import time
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jContentUploader:
    """Upload content to Neo4j knowledge graph."""
    
    def __init__(self, uri, username, password):
        """
        Initialize the Neo4j content uploader.
        
        Args:
            uri: Neo4j URI (e.g., neo4j://128.203.120.208:7687)
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def create_constraint(self, label, property_name):
        """
        Create a uniqueness constraint for a node label and property.
        
        Args:
            label: Node label
            property_name: Property name for the constraint
        """
        with self.driver.session() as session:
            try:
                # Check if constraint exists (Neo4j 4.x syntax)
                result = session.run("SHOW CONSTRAINTS")
                constraints = [record for record in result]
                
                # Check if constraint already exists
                constraint_exists = False
                for constraint in constraints:
                    if label in str(constraint) and property_name in str(constraint):
                        constraint_exists = True
                        break
                
                if not constraint_exists:
                    # Create constraint
                    session.run(
                        f"CREATE CONSTRAINT unique_{label}_{property_name} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.{property_name} IS UNIQUE"
                    )
                    logger.info(f"Created constraint for {label}.{property_name}")
                else:
                    logger.info(f"Constraint for {label}.{property_name} already exists")
            except Exception as e:
                logger.warning(f"Could not create constraint: {str(e)}")
    
    def add_query_answer_pair(self, device_id, query_data):
        """
        Add a query-answer pair to the knowledge graph.
        
        Args:
            device_id: ID of the device associated with the query
            query_data: Query data to add (dict with query, answer, metadata)
        """
        with self.driver.session() as session:
            result = session.execute_write(self._create_query_answer_nodes, device_id, query_data)
            logger.info(f"Added query-answer pair: {result}")
    
    def _create_query_answer_nodes(self, tx, device_id, query_data):
        """
        Create query and answer nodes in the knowledge graph.
        
        Args:
            tx: Neo4j transaction
            device_id: ID of the device
            query_data: Query data to add
        
        Returns:
            Created query ID
        """
        # Generate a query ID if not provided
        if 'id' in query_data:
            query_id = query_data['id']
        else:
            # Create a unique ID based on device and query text
            query_text = query_data['query']['text']
            import hashlib
            hash_object = hashlib.md5(query_text.encode())
            query_id = f"{device_id}_{hash_object.hexdigest()[:8]}"
        
        query_text = query_data['query']['text']
        query_embedding = query_data['query'].get('embedding', [])
        answer_text = query_data['answer']['text']
        answer_embedding = query_data['answer'].get('embedding', [])
        metadata = query_data.get('metadata', {})
        
        # Create Device node
        tx.run(
            "MERGE (d:Device {id: $device_id})",
            device_id=device_id
        )
        
        # Create Query node with embedding
        tx.run(
            "MATCH (d:Device {id: $device_id}) "
            "MERGE (q:Query {id: $query_id}) "
            "SET q.text = $query_text, "
            "    q.embedding = $query_embedding "
            "MERGE (d)-[r:HAS_QUERY]->(q)",
            device_id=device_id, 
            query_id=query_id,
            query_text=query_text,
            query_embedding=query_embedding
        )
        
        # Create Answer node with embedding
        tx.run(
            "MATCH (q:Query {id: $query_id}) "
            "MERGE (a:Answer {id: $query_id + '_answer'}) "
            "SET a.text = $answer_text, "
            "    a.embedding = $answer_embedding "
            "MERGE (q)-[r:HAS_ANSWER]->(a)",
            query_id=query_id,
            answer_text=answer_text,
            answer_embedding=answer_embedding
        )
        
        # Add metadata as properties on the Query node
        for key, value in metadata.items():
            tx.run(
                "MATCH (q:Query {id: $query_id}) "
                "SET q." + key + " = $value",
                query_id=query_id, value=value
            )
        
        return query_id
    
    def add_vector_search_index(self, label, property_name, dimensions):
        """
        Create a vector search index for embeddings.
        
        Args:
            label: Node label (e.g., Query, Answer)
            property_name: Property name for the embeddings
            dimensions: Number of dimensions in the embedding vector
        """
        with self.driver.session() as session:
            try:
                # Check if index exists
                result = session.run("SHOW INDEXES")
                indexes = [record for record in result]
                
                # Create a simple index name
                index_name = f"{label.lower()}_{property_name}_vector_index"
                
                # Check if vector index already exists
                index_exists = False
                for index in indexes:
                    if index_name in str(index):
                        index_exists = True
                        break
                
                if not index_exists:
                    # Create vector index
                    session.run(
                        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                        f"FOR (n:{label}) "
                        f"ON n.{property_name} "
                        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dimensions}, `vector.similarity`: 'cosine' }} }}"
                    )
                    logger.info(f"Created vector index for {label}.{property_name}")
                else:
                    logger.info(f"Vector index for {label}.{property_name} already exists")
            except Exception as e:
                logger.warning(f"Could not create vector index: {str(e)}")
    
    def process_new_format(self, data):
        """
        Process data in the new format from paste.txt.
        
        Args:
            data: Data in the new format with devices and queries
        
        Returns:
            Number of query-answer pairs added
        """
        try:
            # Create constraints
            self.create_constraint('Device', 'id')
            self.create_constraint('Query', 'id')
            self.create_constraint('Answer', 'id')
            
            # Process each device and its queries
            count = 0
            for device in data['devices']:
                device_id = device['id']
                
                for query_data in device['queries']:
                    self.add_query_answer_pair(device_id, query_data)
                    count += 1
            
            # Create vector search indexes (assuming 8 dimensions from the examples)
            embedding_dimensions = len(data['devices'][0]['queries'][0]['query']['embedding'])
            self.add_vector_search_index('Query', 'embedding', embedding_dimensions)
            self.add_vector_search_index('Answer', 'embedding', embedding_dimensions)
            
            return count
        except Exception as e:
            logger.error(f"Error processing new format: {str(e)}")
            raise
    
    def add_data_from_file(self, file_path):
        """
        Add data from a JSON file.
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Number of documents added
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for the new format structure
            if 'devices' in data:
                count = self.process_new_format(data)
                logger.info(f"Successfully added {count} query-answer pairs from {file_path}")
                return count
            else:
                logger.error("File does not contain expected 'devices' structure")
                return 0
            
        except Exception as e:
            logger.error(f"Error adding data from file: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Upload content to Neo4j knowledge graph")
    parser.add_argument("--uri", default="neo4j://128.203.120.208:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="dedee-knowledge-graph!", help="Neo4j password")
    parser.add_argument("--file", required=True, help="JSON file with content to upload")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings for queries and answers")
    
    args = parser.parse_args()
    
    uploader = Neo4jContentUploader(args.uri, args.username, args.password)
    
    try:
        count = uploader.add_data_from_file(args.file)
        logger.info(f"Added {count} query-answer pairs to Neo4j")
    finally:
        uploader.close()

if __name__ == "__main__":
    main()