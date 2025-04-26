# dedee
1. Pushed my docker-retrieval-agent's chromadb container and retrieval-table container to registry
2. Created a deplyoment-template.yaml which specifies the containers in kubernetes pod and exposes their appropriate ports
3. Create a bash script that screates deployments for all the defined regions. Essentially, we create clusters in certain regions
and add a deployment in said cluster (Run kub.deploy.sh)
4. This caused the pods to start running
5. Attached neo4j to each retrieval agent's cluster
6. East US is global one we write to while rest are read-replicas. We also expose the external IP for sync purposes from edge nodes. Work for this is in knowledge-graph/kg-deploy.sh
7. Initialize edge nodes in docker-edge node folder. Create 4 edge device deployments per region.

Data Setup:
- Run get_data.py to extract full csv datasets
- Run edge_info.py to get IP address info and format data into edge data to send. We keep only 5000 from extracted.
    - Split data into 12 splits for each edge device
- Run edge_send.py and individually send to each edge device
- Run build_testset.py and construct a testing data based on data added to edge
    - We sample 500 from data on edge. We augment 70% using LLM to reword, remaining 30% we keep same

Infrastructure setup:
- Created 3 clusters - > edge-eastus, edge-westus, edge-westeurope
- Created 3 clusters -> retrieval-agent-eastus, retrieval-agent-westus, retrieval-agent-west-europe

edge-eastus:
- 4 edge devices
- parent deployment ollama-parent1.yaml (has ollama container)
- deploy judge or judge-parents.yaml here too
- we use default cpu and initialize a gpu pool (2 nodes) and install gpu drivers

edge-westus:
- 4 edge devices
- parent deployment ollama-parent2.yaml (has ollama container)
- we use default cpu and initialize a gpu pool (1 nodes) and install gpu drivers


edge-westeurope:
- 4 edge devices
- parent deployment ollama-parent3.yaml (has ollama container)
- we use default cpu and initialize a gpu pool (1 nodes) and install gpu drivers

retrieval-agent-eastus
- retrieval agent deployment
- neo4j instance

retrieval-agent-eastus
- retrieval agent deployment
- neo4j instance

retrieval-agent-westus
- retrieval agent deployment

retrieval-agent-westeurope
- retrieval agent deployment

Create traffic profile manager that hits health endpoints on each retrieval agent
to route to best retrieval agent

Add models to all ollama models and run testing_main.py to run on specific test set
- Use kubectl CLI to get logs of a given pod's container
Ex. kubectl config use-context edge-eastus
    kubectl get pods
    kubectl logs <pod-name> -c <container-name>


You will also want to set an api_key for helper llm and global_llm in retrieval-table container