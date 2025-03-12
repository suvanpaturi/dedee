from neo4j import GraphDatabase

# Neo4j Connection Details
NEO4J_URI = "bolt://128.203.120.208:7687" # Neo4j Global Graph on eastus
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "dedee-global-knowledge-graph"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def insert_knowledge(tx, record):
    tx.run(
        """
        MERGE (q:Question {text: $query})
        MERGE (a:Answer {text: $response})
        MERGE (s:Source {name: $source})
        MERGE (q)-[:ANSWERED_BY]->(a)
        MERGE (a)-[:SOURCE]->(s)
        """,
        query=record["query"],
        response=record["response"],
        source=record["source"]
    )
# Push Data to Neo4j
send_to_kgraph
with driver.session() as session:
    for record in :
        session.write_transaction(send_to_neo4j, record)

print("Data successfully sent to Neo4j.")
