#!/bin/bash
 
set -e

az login
az acr login --name dedeeregistry

ACR_NAME="dedeeregistry.azurecr.io"

# Versions for each component
JUDGE_VERSION="v2.1"
PARENT_VERSION="v2.5"

echo "🚀 Building JUDGE ($JUDGE_VERSION)..."
docker buildx build --platform linux/amd64 \
  -t $ACR_NAME/judge:latest \
  -t $ACR_NAME/judge:$JUDGE_VERSION \
  --push ./judge
echo "✅ Judge build complete."

echo "🚀 Building PARENT ($PARENT_VERSION)..."
docker buildx build --platform linux/amd64 \
  -t $ACR_NAME/parent:latest \
  -t $ACR_NAME/parent:$PARENT_VERSION \
  --push ./parent
echo "✅ Parent build complete."

echo "🎉 All builds completed and pushed!"
 
echo "Switching to AKS cluster (East US)..."
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing
 
echo "Redeploying Judge..."
kubectl apply -f judge-parents.yaml
 
echo "Redeploying Parent 1..."
kubectl apply -f ollama-parent1.yaml
 
az aks get-credentials --resource-group dedee --name edge-westus --overwrite-existing
echo "Redeploying Parent 2..."
kubectl apply -f ollama-parent2.yaml
 
az aks get-credentials --resource-group dedee --name edge-westeurope --overwrite-existing
 
echo "Redeploying Parent 3..."
kubectl apply -f ollama-parent3.yaml
 
echo "All deployments refreshed!"
 
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing
 
# Check status
kubectl get pods
kubectl get svc