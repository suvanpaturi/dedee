# dedee
1. Pushed my docker-retrieval-agent's chromadb container and retrieval-table container to registry
2. Created a deplyoment-template.yaml which specifies the containers in kubernetes pod and exposes their appropriate ports
3. Create a bash script that screates deployments for all the defined regions. Essentially, we create clusters in certain regions
and add a deployment in said cluster (Run kub.deploy.sh)
4. This caused the pods to start running
5. Created a traffic manager profile in azure and 