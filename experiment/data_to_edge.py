import pandas as pd
import numpy as np
from kubernetes import client, config
import subprocess
import httpx
import asyncio
from neo4j import GraphDatabase

seed = 17
#---------------AWS-------------------#
resource_group = 'dedee'
aks_clusters = [
    {'name': 'edge-eastus', 'resource_group': resource_group},
    {'name': 'edge-westus', 'resource_group': resource_group},
    {'name': 'edge-westeurope', 'resource_group': resource_group}
]
namespace = 'default'

#----------------EDGE DEVICE------------#
DEVICE_PORT = 5001
CLEAR_ENDPOINT = "/clear_kb/"
UPDATE_ENDPOINT = "/update/"
EDGE_SERVICE_IPS = {}

#-------------KNOWLEDGE GRAPH----------------#
NEO4J_URI = "bolt://128.203.120.208:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "dedee-knowledge-graph!"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

#-------------KNOWLEDGE----------------#
#filename = './experiment/data/eli5.csv'
filename = './experiment/data/squad.csv'
#filename = './experiment/data/hotpotqa.csv'

EDGE_KNOWLEDGE_LIST = {}

def get_edge_knowledge(filename):
    dataset = pd.read_csv(filename)
    dataset.columns = ['query', 'response', 'source']
    dataset_shuffle = dataset.sample(frac=0.3, random_state=seed).reset_index(drop=True)
    dataset_splits = np.array_split(dataset_shuffle, 12) #randomly split data across 12 edge nodes
    for i, split in enumerate(dataset_splits):
        edge_knowledge = split.to_dict(orient="records")
        EDGE_KNOWLEDGE_LIST[i] = edge_knowledge

def get_edge_ips():
    for cluster in aks_clusters:
        print(f"\n🔄 Switching to cluster: {cluster['name']}")
        subprocess.run([
            "az", "aks", "get-credentials",
            "--resource-group", cluster["resource_group"],
            "--name", cluster["name"],
            "--overwrite-existing"
        ], check=True)

        config.load_kube_config()
        v1 = client.CoreV1Api()

        services = v1.list_namespaced_service(namespace=namespace).items

        for svc in services:
            if svc.spec.type == "LoadBalancer" and "edge-device" in svc.metadata.name.lower():
                svc_name = svc.metadata.name
                ingress = svc.status.load_balancer.ingress
                ip = ingress[0].ip if ingress else None
                if ip:
                    EDGE_SERVICE_IPS[svc_name] = ip
                else:
                    print(f"⚠️  {svc_name} has no external IP yet.")

def batch(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]
        
async def post_knowledge(device, ip, data):
    url = f"http://{ip}:{DEVICE_PORT}{UPDATE_ENDPOINT}"
    
    async with httpx.AsyncClient() as client:
        try:
           
            '''
            TO BATCH OR NOT TO?
            for i, chunk in enumerate(batch(data, 500)):
                print(f"🚀 Sending {len(chunk)} of batch {i} to {device} → {url}")
                resp = await client.post(url, json={"items": chunk}, timeout=30)
                print()
                print(f"✅ {device} → Status {resp.status_code}")
            '''
            
            print(len(data))
            print(f"🚀 Sending knowledge to {device} → {url}")
            resp = await client.post(url, json={"items": data})
            print(f"✅ {device} → Status {resp.status_code}")
        except httpx.RequestError as e:
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
        except httpx.RequestError as e:
            print(f"❌ {device} → Failed: {str(e)}")
            return False
    return True
            
def clear_knowledge_graph():
    with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ All nodes and relationships have been deleted in knowledge graph")
            
async def distribute_knowledge():
    
    num_devices = len(EDGE_SERVICE_IPS)
    failed_devices = []

    clear_knowledge_graph()

    async def process_device(device, ip, knowledge):
        try:
            await clear_knowledge(device, ip)
            await post_knowledge(device, ip, knowledge)
            print(f"✅ Success for device {device}")
        except (Exception, httpx.RequestError) as e:
            print(f"❌ Failed for device {device}: {str(e)}")
            failed_devices.append(device)

    tasks = [
        process_device(device, ip, EDGE_KNOWLEDGE_LIST[i])
        for i, (device, ip) in enumerate(EDGE_SERVICE_IPS.items())
    ]

    await asyncio.gather(*tasks)

    successful_count = num_devices - len(failed_devices)
    print(f"✅ Successfully distributed dataset {filename} to {successful_count} / {num_devices} edge devices")
    if failed_devices:
        print("❌ The following devices encountered failures:", failed_devices)


if __name__ == "__main__":
    get_edge_ips()
    get_edge_knowledge(filename)
    asyncio.run(distribute_knowledge())

