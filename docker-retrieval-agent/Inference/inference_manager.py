from fastapi import FastAPI
from pydantic import BaseModel
import redis
import os
import json
import time
# from docker_retrieval_agent.retrieval_table.table import RetrievalTable
# from retrieval_table.table import RetrievalTable

app = FastAPI()

# Redis connection
redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

DEBATE_QUEUE = "debate_queue"

# Global table instance
retrieval_table = RetrievalTable(collection_name="retrieval-table")

class Query(BaseModel):
    query: str

@app.post("/inference/")
async def handle_inference(input: Query):
    try:
        # 1. Check retrieval table first
        results = retrieval_table.query(query=input.query)
        if results:
            return {
                "query": input.query,
                "response": results["response"],
                "source": results.get("match_type", "cache")
            }

        # 2. If no match, determine candidate parents
        region_ids = get_relevant_parents(input.query)

        if len(region_ids) == 1:
            redis_client.rpush("debate_pool", f"{region_ids[0]}: {input.query}")
            return {
                "query": input.query,
                "response": "Sent directly to region",
                "region": region_ids[0]
            }

        elif len(region_ids) > 1:
            debate_payload = {
                "query": input.query,
                "region_ids": region_ids,
                "timestamp": time.time()
            }
            redis_client.rpush(DEBATE_QUEUE, json.dumps(debate_payload))
            return {
                "query": input.query,
                "response": "Sent to multi-region debate",
                "participants": region_ids
            }

        else:
            return {
                "query": input.query,
                "response": "No relevant parents found"
            }

    except Exception as e:
        return {"error": str(e)}

def get_relevant_parents(query: str) -> list:
    result = retrieval_table.query(query=query)

    if result and result.get("region"):
        return [result["region"]]  # Return region as a list

    return []  # No relevant region found
