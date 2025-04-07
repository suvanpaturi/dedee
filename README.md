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
- Run edge_info.py to get IP address info and format data into edge data to send
    - Split data into 12 splits for each edge device and add 50% of data in each split into edge device
- Run edge_send.py and individually send to each edge device
- Run build_testset.py and construct a testing data based on data added to edge
    - We sample 500 from data on edge. We keep 70% of this data the exact same, remaining 30% we augment using a LLM
    to reword query 
