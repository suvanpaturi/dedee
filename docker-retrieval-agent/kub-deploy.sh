#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset vars as error

echo "🔧 Building and pushing retrieval-table Docker image..."

docker buildx build --platform linux/amd64 \
  -t dedeeregistry.azurecr.io/retrieval-table:latest \
  -t dedeeregistry.azurecr.io/retrieval-table:v4.2 \
  --push ./retrieval_table

echo "✅ Docker image pushed to dedeeregistry.azurecr.io"

# Define the regions and AKS cluster names
declare -A CLUSTERS
CLUSTERS["eastus"]="retrieval-agent-eastus"
CLUSTERS["westus"]="retrieval-agent-westus"
CLUSTERS["westeurope"]="retrieval-agent-westeurope"

RESOURCE_GROUP="dedee"
ACR_NAME="dedeeregistry"
DEPLOYMENT_TEMPLATE="deployment-template.yaml"

for REGION in "${!CLUSTERS[@]}"; do
  echo "Processing region: $REGION..."

  # Check if AKS cluster exists
  CLUSTER_EXISTS=$(az aks show --resource-group "$RESOURCE_GROUP" --name "${CLUSTERS[$REGION]}" --query "name" --output tsv 2>/dev/null)

  if [[ -z "$CLUSTER_EXISTS" ]]; then
    echo "Creating AKS cluster in $REGION..."
    az aks create --resource-group "$RESOURCE_GROUP" \
      --name "${CLUSTERS[$REGION]}" \
      --node-count 2 \
      --enable-managed-identity \
      --enable-addons monitoring \
      --location "$REGION" \
      --generate-ssh-keys
      echo "Cluster ${CLUSTERS[$REGION]} successfully created"
  else
    echo "AKS cluster in $REGION already exists. Skipping creation."
  fi

  # Attach ACR to the AKS cluster (safe to run multiple times)
  echo "Attaching ACR to AKS in $REGION..."
  az aks update --name "${CLUSTERS[$REGION]}" --resource-group "$RESOURCE_GROUP" --attach-acr $ACR_NAME

  # Get credentials for the correct AKS cluster
  echo "Fetching AKS credentials..."
  az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "${CLUSTERS[$REGION]}"

  # Generate deployment file for this region
  echo "Generating deployment YAML for $REGION..."
  sed "s/{{REGION}}/$REGION/g" "$DEPLOYMENT_TEMPLATE" > "deployment-$REGION.yaml"

  # Apply the Kubernetes deployment
  echo "Deploying workloads to AKS in $REGION..."
  kubectl apply -f "deployment-$REGION.yaml"

  echo "Deployment to $REGION completed."
done

echo "All regions deployed successfully!"