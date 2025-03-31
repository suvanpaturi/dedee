#!/bin/bash

# Define the regions and AKS cluster names
declare -A CLUSTERS
CLUSTERS["eastus"]="edge-eastus"
CLUSTERS["westus"]="edge-westus"
CLUSTERS["westeurope"]="edge-westeurope"

RESOURCE_GROUP="dedee"
ACR_NAME="dedeeregistry"
DEPLOYMENT_TEMPLATE="deployment-template.yaml"
DEPLOYMENT_ROOT="deployments"

# Loop through each region
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
  az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "${CLUSTERS[$REGION]}" --overwrite-existing

  # Create a subfolder for this region inside deployments/
  REGION_FOLDER="$DEPLOYMENT_ROOT/$REGION"
  mkdir -p "$REGION_FOLDER"

  # Deploy 4 unique edge devices per region
  for DEVICE_ID in {1..4}; do
    DEPLOYMENT_FILE="$REGION_FOLDER/edge-device-$DEVICE_ID.yaml"

    # Generate region-specific deployment file and store it in the correct folder
    sed "s/{{REGION}}/$REGION/g; s/{{DEVICE_ID}}/$DEVICE_ID/g" "$DEPLOYMENT_TEMPLATE" > "$DEPLOYMENT_FILE"

    # Apply the deployment
    kubectl apply -f "$DEPLOYMENT_FILE"

    echo "Deployed edge-device-$DEVICE_ID in $REGION (Stored in $DEPLOYMENT_FILE)"
  done

  echo "Deployment to $REGION completed."
done

echo "All regions deployed successfully!"
