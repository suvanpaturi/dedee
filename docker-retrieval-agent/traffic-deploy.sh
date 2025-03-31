#!/bin/bash
# Bash file to setup DNS Traffic Manager to route queries to closest retrieval agent

# Configuration
RESOURCE_GROUP="dedee"
DNS_PREFIX="retrieval-agent-traffic"
PORT="5001"
HEALTH_PATH="/ping/"

# Usage check
if [ -z "$RESOURCE_GROUP" ] || [ -z "$DNS_PREFIX" ]; then
    echo "Please edit this script to set RESOURCE_GROUP and DNS_PREFIX variables."
    echo "Example:"
    echo "  RESOURCE_GROUP=\"mygroup\""
    echo "  DNS_PREFIX=\"myretrievalagents\""
    exit 1
fi

# Check Azure login
if ! az account show >/dev/null 2>&1; then
    echo "Error: You must be logged in to Azure CLI. Run 'az login' first."
    exit 1
fi

# Create Traffic Manager profile
echo "Creating Traffic Manager profile..."
az network traffic-manager profile create \
    --name "retrieval-tm-profile" \
    --resource-group "$RESOURCE_GROUP" \
    --routing-method Performance \
    --unique-dns-name "$DNS_PREFIX" \
    --ttl 30 \
    --protocol HTTP \
    --port "$PORT" \
    --path "$HEALTH_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create Traffic Manager profile."
    exit 1
fi

# Function to add an endpoint
add_endpoint() {
    local name=$1
    local ip=$2
    local location=$3
    
    echo "Adding endpoint: $name ($ip) in $location..."
    az network traffic-manager endpoint create \
        --resource-group "$RESOURCE_GROUP" \
        --profile-name "retrieval-tm-profile" \
        --name "$name" \
        --type externalEndpoints \
        --target "$ip" \
        --endpoint-status Enabled \
        --endpoint-location "$location"
}

# Add endpoints (uncomment and edit these lines as needed)

add_endpoint "retrieval-agent-eastus" "28.203.120.105" "eastus"
add_endpoint "retrieval-agent-westeurope" "9.163.200.77" "westeurope"
add_endpoint "retrieval-agent-westus" "52.159.147.210" "westus"

# Get the Traffic Manager URL
TRAFFIC_MANAGER_URL="${DNS_PREFIX}.trafficmanager.net"

# Summary
echo ""
echo "Traffic Manager Setup Complete"
echo "------------------------------"
echo "URL: http://${TRAFFIC_MANAGER_URL}:${PORT}"
echo "Health Path: ${HEALTH_PATH}"
echo ""
echo "To add further endpoints, run:"
echo "az network traffic-manager endpoint create \\"
echo "  --resource-group \"$RESOURCE_GROUP\" \\"
echo "  --profile-name \"retrieval-tm-profile\" \\"
echo "  --name \"region-name\" \\"
echo "  --type externalEndpoints \\"
echo "  --target \"IP_ADDRESS\" \\"
echo "  --endpoint-status Enabled \\"
echo "  --endpoint-location \"azure-region\""
echo ""
echo "Python usage example:"
echo "import requests"
echo "BASE_URL = \"http://${TRAFFIC_MANAGER_URL}:${PORT}\""
echo "response = requests.get(f\"{BASE_URL}/get/\", params={\"query\": \"hello\"})"
echo "print(response.json())"