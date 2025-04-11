#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset vars as error

NAMESPACE=default
MODEL_NAME="gemma:2b"

delete_and_pull_model() {
  local pod_name=$1
  local role_name=$2

  if [ -n "$pod_name" ]; then
    echo "Checking and deleting existing model for $role_name..."
    kubectl exec "$pod_name" -- ollama rm "$MODEL_NAME" || echo "Model not found, skipping delete"

    echo "Pulling latest model for $role_name..."
    kubectl exec "$pod_name" -- ollama pull "$MODEL_NAME"
  else
    echo "No $role_name pod found. Skipping."
  fi
}

echo "Switching to AKS cluster (East US)..."
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing

echo "Updating Judge..."
JUDGE_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep judge | awk '{print $1}')
delete_and_pull_model "$JUDGE_POD" "Judge"

echo "Updating Parent (East US)..."
PARENT_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
delete_and_pull_model "$PARENT_POD" "Parent-EastUS"

echo "Switching to AKS cluster (West US)..."
az aks get-credentials --resource-group dedee --name edge-westus --overwrite-existing

echo "Updating Parent (West US)..."
PARENT_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
delete_and_pull_model "$PARENT_POD" "Parent-WestUS"

echo "Switching to AKS cluster (West Europe)..."
az aks get-credentials --resource-group dedee --name edge-westeurope --overwrite-existing

echo "Updating Parent (West Europe)..."
PARENT_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
delete_and_pull_model "$PARENT_POD" "Parent-WestEurope"

echo "✅ Pulled and refreshed model across all parents and judge."

# Return to East US context and show pod/svc status
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing
kubectl get pods
kubectl get svc
