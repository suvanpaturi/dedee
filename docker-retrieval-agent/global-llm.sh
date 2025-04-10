#!/bin/bash
 
set -e  # Exit if any command fails
set -u  # Treat unset variables as errors
 
NAMESPACE=default

echo "Adding secret to east us retrieval agent"
az aks get-credentials --resource-group dedee --name retrieval-agent-eastus --overwrite-existing
kubectl delete secret openai-secret
kubectl create secret generic openai-secret --from-env-file=.env

echo "Adding secret to west us retrieval agent"
az aks get-credentials --resource-group dedee --name retrieval-agent-westus --overwrite-existing
kubectl delete secret openai-secret
kubectl create secret generic openai-secret --from-env-file=.env

echo "Adding secret to west europe retrieval agent"
az aks get-credentials --resource-group dedee --name retrieval-agent-westeurope --overwrite-existing
kubectl delete secret openai-secret
kubectl create secret generic openai-secret --from-env-file=.env
