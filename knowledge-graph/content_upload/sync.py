import requests
import logging
import json
from neo4j import GraphDatabase
from typing import List
from knowledge_base import Knowledge
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://128.203.120.208:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dedee-knowledge-graph!")
DEVICE_ID = os.getenv("DEVICE_ID", "edge-device-1")

class Neo4jUploader:
    """Class to handle uploading content to Neo4j from edge devices."""
    
    def __init__(self, uri, username, password, device_id):
        """
        Initialize the Neo4j uploader.
        
        Args:
            uri: Neo4j URI (e.g., neo4j://128.203.120.208:7687)
            username: Neo4j username
            password: Neo4j password
            device_id: Identifier for this edge device
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.device_id = device_id
        logger.info(f"Connected to Neo4j at {uri} from device {device_id}")
    
    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def upload_knowledge_items(self, items: List[Knowledge]) -> int:
        """
        Upload knowledge items to Neo4j.
        
        Args:
            items: List of Knowledge items to upload
            
        Returns:
            Number of items successfully uploaded
        """
        successful_count = 0
        
        try:
            with self.driver.session() as session:
                # First create or update the device node
                session.run(
                    "MERGE (d:Device {id: $device_id}) "
                    "SET d.last_sync = $timestamp",
                    device_id=self.device_id,
                    timestamp=datetime.now().isoformat()
                )
                
                # Upload each knowledge item
                for item in items:
                    try:
                        # Create query with embedding
                        query_id = self._create_unique_id(item.query)
                        
                        # Upload query and answer
                        result = session.run(
                            """
                            // Create or update query node
                            MERGE (q:Query {id: $query_id})
                            SET q.text = $query_text,
                                q.embedding = $query_embedding
                            
                            // Create or update answer node
                            MERGE (a:Answer {id: $query_id + '_answer'})
                            SET a.text = $answer_text,
                                a.embedding = $answer_embedding
                            
                            // Create relationships
                            MERGE (d:Device {id: $device_id})
                            MERGE (d)-[r1:HAS_QUERY]->(q)
                            MERGE (q)-[r2:HAS_ANSWER]->(a)
                            
                            // Add metadata
                            SET q.source = $source,
                                q.sync_timestamp = $timestamp
                                
                            RETURN q.id as query_id
                            """,
                            query_id=query_id,
                            query_text=item.query,
                            query_embedding=item.query_embedding,
                            answer_text=item.response,
                            answer_embedding=item.response_embedding,
                            device_id=self.device_id,
                            source=f"edge-device-{self.device_id}",
                            timestamp=datetime.now().isoformat()
                        )
                        
                        record = result.single()
                        if record:
                            logger.info(f"Uploaded knowledge item with ID: {record['query_id']}")
                            successful_count += 1
                    except Exception as e:
                        logger.error(f"Error uploading item {item.query[:30]}...: {str(e)}")
            
            return successful_count
        except Exception as e:
            logger.error(f"Error in bulk upload: {str(e)}")
            return successful_count
    
    def _create_unique_id(self, text: str) -> str:
        """Create a unique ID based on device ID and text content."""
        import hashlib
        hash_object = hashlib.md5(text.encode())
        return f"{self.device_id}_{hash_object.hexdigest()[:8]}"


def send_to_kgraph(items: List[Knowledge]) -> bool:
    """
    Send knowledge items to the Neo4j knowledge graph.
    
    Args:
        items: List of Knowledge items to send
        
    Returns:
        True if successful, False otherwise
    """
    uploader = Neo4jUploader(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        device_id=DEVICE_ID
    )
    
    try:
        count = uploader.upload_knowledge_items(items)
        logger.info(f"Successfully uploaded {count} out of {len(items)} items to Neo4j")
        return count > 0
    except Exception as e:
        logger.error(f"Error sending to knowledge graph: {str(e)}")
        return False
    finally:
        uploader.close()


if __name__ == "__main__":
    # This allows the script to be run directly for testing or manual syncing
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync knowledge items to Neo4j")
    parser.add_argument("--file", help="JSON file with knowledge items to sync")
    parser.add_argument("--device", default=DEVICE_ID, help="Device ID to use")
    
    args = parser.parse_args()
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                data = json.load(f)
            
            # Convert to Knowledge objects
            items = []
            for item in data:
                if 'query' in item and 'response' in item:
                    knowledge = Knowledge(
                        query=item['query'],
                        response=item['response'],
                        query_embedding=item.get('query_embedding', None),
                        response_embedding=item.get('response_embedding', None)
                    )
                    items.append(knowledge)
            
            # Set device ID from args
            DEVICE_ID = args.device
            
            # Send to Neo4j
            success = send_to_kgraph(items)
            if success:
                print(f"Successfully synced items from {args.file} to Neo4j")
            else:
                print(f"Failed to sync items from {args.file} to Neo4j")
        except Exception as e:
            print(f"Error processing file: {str(e)}")