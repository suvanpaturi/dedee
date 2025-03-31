from sync.graph_manager import GraphManager
from sync.graph_embedder import GraphEmbedder
from typing import List, Dict
import json
import os

EDGE_ID = os.getenv("EDGE_ID", "default-id") 
EDGE_REGION = os.getenv("EDGE_REGION", "default-region")

class SyncManager():
    def __init__(self):
        self.graph_manager = GraphManager()
        self.embedder = GraphEmbedder("all-MiniLM-L6-v2")
        
    def send_knowledge(self, added_data: List[Dict[str, str]]):
        transformed_data = self.embedder.transform_edge_knowledge(added_data, EDGE_ID, EDGE_REGION)
        self.graph_manager.import_data(json.dumps(transformed_data, indent=4))