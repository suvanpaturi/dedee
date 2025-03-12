#!/bin/bash

# Neo4j Azure Kubernetes Cluster Installation and Replication
set -e

PRIMARY_CLUSTER="retrieval-agent-eastus"
REPLICA_CLUSTERS=("retrieval-agent-westus" "retrieval-agent-westeurope")
RESOURCE_GROUP="dedee"
NAMESPACE="default"
NEO4J_HELM_CHART_VERSION="5.10.0"
NEO4J_PASSWORD="dedee-knowledge-graph!"

for cmd in az kubectl helm; do
  if ! command -v $cmd &> /dev/null; then
    case $cmd in
      az) curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash ;;
      kubectl) sudo az aks install-cli ;;
      helm) curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash ;;
    esac
  fi
done

# Function to connect to a cluster
connect_to_cluster() {
  echo "Connecting to cluster: $1"
  az aks get-credentials --resource-group $RESOURCE_GROUP --name $1 --overwrite-existing
}

# Function to check if a cluster exists
check_cluster_exists() {
  az aks show --name $1 --resource-group $RESOURCE_GROUP &> /dev/null || 
    { echo "Error: Cluster $1 does not exist in resource group $RESOURCE_GROUP"; return 1; }
}

# Function to install Neo4j using Helm
install_neo4j() {
  local cluster_name=$1
  local role=$2  # "primary" or "replica"
  
  connect_to_cluster $cluster_name

  # Ensure Helm is up-to-date
  helm repo add neo4j https://helm.neo4j.com/neo4j
  helm repo update

  # Create values file
  if [ "$role" = "primary" ]; then
    cat > neo4j-values.yaml <<EOF
neo4j:
  name: neo4j-$cluster_name
  password: "$NEO4J_PASSWORD"
  acceptLicenseAgreement: "yes"
  edition: enterprise
  resources:
    requests:
        memory: "2Gi"
        cpu: "500m"
    limits:
        memory: "3Gi"
        cpu: "1"
  storage:
    storageClassName: "managed-premium"
    volumeSize: "100Gi"

volumes:
  data:
    mode: defaultStorageClass

cluster:
  mode: standalone
  minimumClusterSize: 1

services:
  neo4j:
    enabled: true
    type: LoadBalancer
EOF
  else
    # Use the stored global primary IP instead of switching contexts
    PRIMARY_IP=$GLOBAL_PRIMARY_IP
    echo "Using primary IP from global variable: $PRIMARY_IP"
    
    cat > neo4j-values.yaml <<EOF
neo4j:
  name: neo4j-$cluster_name
  password: "$NEO4J_PASSWORD"
  acceptLicenseAgreement: "yes"
  edition: enterprise
  resources:
    requests:
        memory: "2Gi"
        cpu: "500m"
    limits:
        memory: "3Gi"
        cpu: "1"
  storage:
    storageClassName: "managed-premium"
    volumeSize: "100Gi"

volumes:
  data:
    mode: defaultStorageClass

cluster:
  mode: readReplica
  minimumClusterSize: 1

readReplica:
  primaryServer: "neo4j://$PRIMARY_IP:7687"

services:
  neo4j:
    enabled: true
    type: LoadBalancer
EOF
  fi

  # Install Neo4j with Helm
  echo "Installing Neo4j on $cluster_name as $role..."
  if ! helm upgrade --install neo4j-$cluster_name neo4j/neo4j \
    --namespace default \
    --version $NEO4J_HELM_CHART_VERSION \
    --set services.neo4j.type=LoadBalancer \
    -f neo4j-values.yaml; then
    echo "❌ Helm installation failed on $cluster_name. Exiting."
    exit 1
  fi

  # Wait for Neo4j pods to be ready (using correct labels)
  echo "Waiting for Neo4j pods to be ready on $cluster_name..."
  sleep 30
  if ! kubectl wait --for=condition=ready pod -l app=neo4j-$cluster_name --timeout=500s -n default; then
    echo "❌ Neo4j pods failed to start on $cluster_name. Check logs."
    kubectl logs -l app=neo4j-$cluster_name -n default
    exit 1
  fi

  # Display service information
  kubectl get svc -n default neo4j-$cluster_name -o wide || true
  
  # Check if service has an external IP
  if [ -z "$(kubectl get svc -n default neo4j-$cluster_name -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)" ]; then
    echo "No external IP found. Patching service to LoadBalancer type..."
    kubectl patch svc -n default neo4j-$cluster_name -p '{"spec":{"type":"LoadBalancer"}}'
    echo "Waiting 30 seconds for external IP assignment..."
    sleep 30
    kubectl get svc -n default neo4j-$cluster_name -o wide || true
  fi
}

# Function to check replication status
check_replication_status() {
  connect_to_cluster $1
  POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=neo4j-$1 -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -z "$POD_NAME" ]; then
    echo "No pods found for neo4j-$1. Skipping replication check."
    return
  fi
  echo "Checking replication status on $1..."
  kubectl exec -n $NAMESPACE $POD_NAME -- bash -c "echo 'SHOW DATABASES;' | cypher-shell -u neo4j -p \"$NEO4J_PASSWORD\"" || echo "Could not check replication status on $1"
}

# Check if all clusters exist
for cluster in $PRIMARY_CLUSTER "${REPLICA_CLUSTERS[@]}"; do
  check_cluster_exists $cluster || exit 1
done

# Install Neo4j on primary cluster
echo "=== Installing Neo4j on primary cluster ==="
install_neo4j $PRIMARY_CLUSTER "primary"

# Connect to primary to get its IP
connect_to_cluster $PRIMARY_CLUSTER
PRIMARY_IP=$(kubectl get svc -n $NAMESPACE neo4j-$PRIMARY_CLUSTER -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
if [ -z "$PRIMARY_IP" ]; then
  echo "Could not get primary IP. The LoadBalancer may still be provisioning."
  echo "Waiting 60 seconds for IP address assignment..."
  sleep 60
  PRIMARY_IP=$(kubectl get svc -n $NAMESPACE neo4j-$PRIMARY_CLUSTER -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
  if [ -z "$PRIMARY_IP" ]; then
    echo "Still could not get primary IP. Please check the service status manually."
    kubectl get svc -n $NAMESPACE neo4j-$PRIMARY_CLUSTER
    exit 1
  fi
fi
echo "Primary Neo4j instance IP: $PRIMARY_IP"

# Make PRIMARY_IP variable available to the rest of the script
export PRIMARY_NEO4J_IP="$PRIMARY_IP"

# Store PRIMARY_IP in a global variable to ensure it's available
export GLOBAL_PRIMARY_IP=$PRIMARY_IP

# Install Neo4j replicas on other clusters
for replica in "${REPLICA_CLUSTERS[@]}"; do
  echo "=== Installing Neo4j replica on $replica ==="
  install_neo4j $replica "replica"
done

# Wait for replication initialization
echo "Waiting for replication to initialize (60 seconds)..."
sleep 60

# Print summary of IP addresses
echo "=== Neo4j Installation IP Summary ==="
connect_to_cluster $PRIMARY_CLUSTER
PRIMARY_EXTERNAL_IP=$(kubectl get svc -n $NAMESPACE neo4j-$PRIMARY_CLUSTER -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
echo "Primary Neo4j instance IP: $PRIMARY_EXTERNAL_IP"

for replica in "${REPLICA_CLUSTERS[@]}"; do
  connect_to_cluster $replica
  echo "Replica ($replica): $(kubectl get svc -n $NAMESPACE neo4j-$replica -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "IP not available")"
done
echo ""
echo "Neo4j installation and replication completed!"