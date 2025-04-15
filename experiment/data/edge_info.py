import pandas as pd
import numpy as np
from kubernetes import client, config
import subprocess
import json

seed = 17

#---------------AWS-------------------#
resource_group = 'dedee'
aks_clusters = [
    {'name': 'edge-eastus', 'resource_group': resource_group},
    {'name': 'edge-westus', 'resource_group': resource_group},
    {'name': 'edge-westeurope', 'resource_group': resource_group}
]
namespace = 'default'

#-------------KNOWLEDGE----------------#
#filename = './experiment/data/eli5.csv'
#filename = './experiment/data/squad.csv'
filename = './experiment/data/extracted/hotpotqa.csv'

EDGE_MAPPING = {}
EDGE_SERVICE_IPS = {}

def get_edge_knowledge(filename):
    dataset = pd.read_csv(filename)
    dataset.columns = ['query', 'response', 'source']
    dataset_shuffle = dataset.sample(n=min(len(dataset), 5000), random_state=seed).reset_index(drop=True)
    dataset_splits = np.array_split(dataset_shuffle, 12)
    '''
    dataset_splits = [
        split.sample(frac=0.5, random_state=seed)
        for split in dataset_splits
    ]
    '''
    for i, split in enumerate(dataset_splits):
        edge_knowledge = split.to_dict(orient="records")
        with open(f"./experiment/data/edge/{EDGE_MAPPING[i]}.json", "w") as f:
            json.dump(edge_knowledge, f, indent=4)

def get_edge_ips():
    num_devices = 0
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
                    EDGE_MAPPING[num_devices] = svc_name
                    num_devices +=1
                else:
                    print(f"⚠️  {svc_name} has no external IP yet.")
                    
        with open(f"./experiment/data/edge/device_ips.json", "w") as f:
            json.dump(EDGE_SERVICE_IPS, f, indent=4)
                       
if __name__ == "__main__":
    get_edge_ips()
    get_edge_knowledge(filename)
    

