import argparse
import logging
from neo4j import GraphDatabase

# python delete.py --all
# python delete.py --device laptop
# python delete.py --query laptop_12345678

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jContentDeleter:
    """Delete content from Neo4j knowledge graph."""
    
    def __init__(self, uri, username, password):
        """
        Initialize the Neo4j content deleter.
        
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
    
    def delete_all_data(self):
        """Delete all data from the knowledge graph."""
        with self.driver.session() as session:
            try:
                # Delete all relationships and nodes
                query = "MATCH (n) DETACH DELETE n"
                result = session.run(query)
                return result.consume().counters
            except Exception as e:
                logger.error(f"Error deleting all data: {str(e)}")
                raise
    
    def delete_device_data(self, device_id):
        """
        Delete data for a specific device.
        
        Args:
            device_id: ID of the device to delete
        """
        with self.driver.session() as session:
            try:
                # First find all queries belonging to this device
                query = (
                    "MATCH (d:Device {id: $device_id})-[:HAS_QUERY]->(q:Query) "
                    "RETURN q.id as query_id"
                )
                result = session.run(query, device_id=device_id)
                query_ids = [record["query_id"] for record in result]
                
                # Delete answers for these queries
                for query_id in query_ids:
                    answer_query = (
                        "MATCH (q:Query {id: $query_id})-[:HAS_ANSWER]->(a:Answer) "
                        "DETACH DELETE a"
                    )
                    session.run(answer_query, query_id=query_id)
                
                # Delete queries
                queries_query = (
                    "MATCH (d:Device {id: $device_id})-[:HAS_QUERY]->(q:Query) "
                    "DETACH DELETE q"
                )
                session.run(queries_query, device_id=device_id)
                
                # Finally delete the device
                device_query = "MATCH (d:Device {id: $device_id}) DETACH DELETE d"
                result = session.run(device_query, device_id=device_id)
                
                return result.consume().counters
            except Exception as e:
                logger.error(f"Error deleting device data: {str(e)}")
                raise
    
    def delete_query(self, query_id):
        """
        Delete a specific query and its answer.
        
        Args:
            query_id: ID of the query to delete
        """
        with self.driver.session() as session:
            try:
                # Delete the answer
                answer_query = (
                    "MATCH (q:Query {id: $query_id})-[:HAS_ANSWER]->(a:Answer) "
                    "DETACH DELETE a"
                )
                session.run(answer_query, query_id=query_id)
                
                # Delete the query
                query = "MATCH (q:Query {id: $query_id}) DETACH DELETE q"
                result = session.run(query, query_id=query_id)
                
                return result.consume().counters
            except Exception as e:
                logger.error(f"Error deleting query: {str(e)}")
                raise

def main():
    parser = argparse.ArgumentParser(description="Delete content from Neo4j knowledge graph")
    parser.add_argument("--uri", default="neo4j://128.203.120.208:7687", help="Neo4j URI")
    parser.add_argument("--username", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="dedee-knowledge-graph!", help="Neo4j password")
    parser.add_argument("--all", action="store_true", help="Delete all data")
    parser.add_argument("--device", help="Delete data for a specific device ID")
    parser.add_argument("--query", help="Delete a specific query ID")
    
    args = parser.parse_args()
    
    if not (args.all or args.device or args.query):
        parser.error("No action specified. Use --all, --device, or --query.")
    
    deleter = Neo4jContentDeleter(args.uri, args.username, args.password)
    
    try:
        if args.all:
            logger.info("Deleting all data...")
            counters = deleter.delete_all_data()
            logger.info(f"Deleted: {counters}")
        elif args.device:
            logger.info(f"Deleting data for device: {args.device}")
            counters = deleter.delete_device_data(args.device)
            logger.info(f"Deleted: {counters}")
        elif args.query:
            logger.info(f"Deleting query: {args.query}")
            counters = deleter.delete_query(args.query)
            logger.info(f"Deleted: {counters}")
    finally:
        deleter.close()

if __name__ == "__main__":
    main()