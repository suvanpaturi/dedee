#!/bin/bash
 
set -e  # Exit if any command fails
set -u  # Treat unset variables as errors
 
NAMESPACE=default

echo "Switching to AKS cluster (East US)..."
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing
 
echo "Pulling for Judge..."

POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep judge | awk '{print $1}')
if [ -n "$POD" ]; then
    echo "Pulling latest model for Judge..."
    kubectl exec $POD -- ollama pull gemma:2b
else
    echo "No parent pod found. Skipping model pull."
fi

echo "Pulling for Parent.."

POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
if [ -n "$POD" ]; then
    echo "Pulling latest model for Parent..."
    kubectl exec $POD -- ollama pull gemma:2b
else
    echo "No parent pod found. Skipping model pull."
fi
 
az aks get-credentials --resource-group dedee --name edge-westus --overwrite-existing

echo "Pulling for Parent.."

POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
if [ -n "$POD" ]; then
    echo "Pulling latest model for Parent..."
    kubectl exec $POD -- ollama pull gemma:2b
else
    echo "No parent pod found. Skipping model pull."
fi
 
az aks get-credentials --resource-group dedee --name edge-westeurope --overwrite-existing
 
echo "Pulling for Parent.."

POD=$(kubectl get pods --namespace $NAMESPACE --no-headers | grep parent | awk '{print $1}')
if [ -n "$POD" ]; then
    echo "Pulling latest model for Parent..."
    kubectl exec $POD -- ollama pull gemma:2b
else
    echo "No parent pod found. Skipping model pull."
fi
 
echo "Pulled latest model across all parents and judge"
 
az aks get-credentials --resource-group dedee --name edge-eastus --overwrite-existing
 
# Check status
kubectl get pods
kubectl get svc