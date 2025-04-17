#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset vars as error

NAMESPACE=default

# Parent model
MODEL_NAME_TO_REMOVE="qwen:0.5b"
MODEL_NAME="qwen:0.5b"

# Judge model
JUDGE_MODEL_TO_REMOVE="gemma:2b"
JUDGE_MODEL="gemma:2b"

delete_and_pull_model() {
  local pod_name=$1
  local role_name=$2
  local model_to_remove=$3
  local model_to_pull=$4

  if [ -n "$pod_name" ]; then
    echo "Checking and deleting existing model ($model_to_remove) for $role_name..."
    kubectl exec "$pod_name" -- ollama rm "$model_to_remove" || echo "Model not found, skipping delete"

    echo "Pulling latest model ($model_to_pull) for $role_name..."
    kubectl exec "$pod_name" -- ollama pull "$model_to_pull"
  else
    echo "No $role_name pod found. Skipping."
  fi
}

echo "Switching to AKS cluster (East US)..."
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing

echo "Updating Judge..."
JUDGE_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep judge | awk '{print $1}')
delete_and_pull_model "$JUDGE_POD" "Judge" "$JUDGE_MODEL_TO_REMOVE" "$JUDGE_MODEL"

echo "Updating Parent (East US)..."
PARENT_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
delete_and_pull_model "$PARENT_POD" "Parent-EastUS" "$MODEL_NAME_TO_REMOVE" "$MODEL_NAME"

echo "Switching to AKS cluster (West US)..."
az aks get-credentials --resource-group dedee --name edge-westus --overwrite-existing

echo "Updating Parent (West US)..."
PARENT_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
delete_and_pull_model "$PARENT_POD" "Parent-WestUS" "$MODEL_NAME_TO_REMOVE" "$MODEL_NAME"

echo "Switching to AKS cluster (West Europe)..."
az aks get-credentials --resource-group dedee --name edge-westeurope --overwrite-existing

echo "Updating Parent (West Europe)..."
PARENT_POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
delete_and_pull_model "$PARENT_POD" "Parent-WestEurope" "$MODEL_NAME_TO_REMOVE" "$MODEL_NAME"

echo "✅ Pulled and refreshed model across all parents and judge."

# Return to East US context and show pod/svc status
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing
kubectl get pods
kubectl get svc
