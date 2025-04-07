import time
import httpx
import asyncio
import json
from neo4j import GraphDatabase

#----------------EDGE DEVICE------------#
DEVICE_PORT = 5001
CLEAR_ENDPOINT = "/clear_kb/"
UPDATE_ENDPOINT = "/update/"

#-------------KNOWLEDGE GRAPH----------------#
NEO4J_URI = "bolt://128.203.120.208:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "dedee-knowledge-graph!"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

async def post_knowledge(device, ip, data):
    url = f"http://{ip}:{DEVICE_PORT}{UPDATE_ENDPOINT}"
    
    async with httpx.AsyncClient() as client:
        try:
            print(len(data))
            print(f"🚀 Sending knowledge to {device} → {url}")
            resp = await client.post(url, json={"items": data})
            print(f"✅ {device} → Status {resp.status_code}")
        except Exception as e:
            print(f"❌ {device} → Failed: {str(e)}")
            return False
    return True
            
async def clear_knowledge(device, ip):
    url = f"http://{ip}:{DEVICE_PORT}{CLEAR_ENDPOINT}"
    async with httpx.AsyncClient() as client:
        try:
            print(f"🧹 Clearing {device} → {url}")
            resp = await client.post(url, timeout=20.0)
            print(f"✅ {device} → Status {resp.status_code}")
        except Exception as e:
            print(f"❌ {device} → Failed: {str(e)}")
            return False
    return True
            
def clear_orphaned_nodes():
     with driver.session() as session:
            session.run("MATCH (n) WHERE NOT (n)-[*]-(:Device) DELETE n")
            print("🗑️ Deleted nodes and relationships not tied to a specific device")
            
def clear_knowledge_graph():
    with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ All nodes and relationships have been deleted in knowledge graph")

def device_exists(device):
    region, device_id = device.replace('edge-device-', "").replace("-service", "").split("-")
    query = """
    MATCH (d:Device {device_id: $device_id, device_region: $region})
    OPTIONAL MATCH (d)-[:HAS_QUERY]->(q:Query)
    RETURN count(q) AS queryCount, count(d) > 0 AS exists
    """
    with driver.session() as session:
        result = session.run(query, device_id=device_id, region=region).single()

    exists = result["exists"]
    query_count = result["queryCount"] if exists else 0

    print("Device exists:", exists)
    print("Query connections:", query_count)

async def process_device(device):
    try:
        with open(f"./experiment/data/edge/{device}.json", "r") as f1:
            knowledge = json.load(f1)
        with open(f"./experiment/data/edge/device_ips.json", "r") as f2:
            ips = json.load(f2)
        await clear_knowledge(device, ips[device])
        await post_knowledge(device, ips[device], knowledge)
        print(f"✅ Successfully triggered    data deployment for device {device}")
    except (Exception) as e:
        print(f"❌ Failed for device {device}: {str(e)}")
        
        
if __name__ == "__main__":
    #run once
    #clear_knowledge_graph()
    #run per device
        
    #asyncio.run(process_device("edge-device-eastus-1-service"))
    #asyncio.run(process_device("edge-device-eastus-2-service"))
    #asyncio.run(process_device("edge-device-eastus-3-service"))
    #asyncio.run(process_device("edge-device-eastus-4-service"))

    #asyncio.run(process_device("edge-device-westus-1-service"))
    #asyncio.run(process_device("edge-device-westus-2-service"))
    #asyncio.run(process_device("edge-device-westus-3-service"))
    #asyncio.run(process_device("edge-device-westus-4-service"))
    
    #asyncio.run(process_device("edge-device-westeurope-1-service"))
    #asyncio.run(process_device("edge-device-westeurope-2-service"))
    #asyncio.run(process_device("edge-device-westeurope-3-service"))
    #asyncio.run(process_device("edge-device-westeurope-4-service"))
    
    
    device_exists(device="edge-device-eastus-4-service")
