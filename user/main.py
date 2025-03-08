from models.query import UserQuery
import requests
from azure.identity import AzureCliCredential
from azure.mgmt.containerservice import ContainerServiceClient
import subprocess


#query = UserQuery(user='suvpat', query='Who is the president of the United States?')
#response = requests.post("http://localhost:8000/query/", json={"query": query.query})


# Authenticate using Azure CLI credentials
credential = AzureCliCredential()
subscription_id = 'e1f296d3-09bc-4d6f-8786-95a3675d78f7'

# Initialize the AKS client
aks_client = ContainerServiceClient(credential, subscription_id)

# List all AKS clusters
clusters = aks_client.managed_clusters.list()

# Extract cluster details
cluster_info = []
for cluster in clusters:
    cluster_info.append({
        'name': cluster.name,
        'location': cluster.location,
        'fqdn': cluster.fqdn  # Fully Qualified Domain Name of the API server
    })

def ping_server(server, count=3):
    try:
        output = subprocess.run(
            ["ping", "-c", str(count), server],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        # Extract the average RTT from the ping output
        for line in output.stdout.split('\n'):
            if 'avg' in line:
                avg_rtt = float(line.split('/')[4])
                return avg_rtt
    except Exception as e:
        print(f"Error pinging {server}: {e}")
    return float('inf')  # Return a high value if ping fails

# Measure latency to each cluster
for cluster in cluster_info:
    latency = ping_server(cluster['fqdn'])
    cluster['latency'] = latency
    print(f"Cluster: {cluster['name']}, Location: {cluster['location']}, Latency: {latency} ms")