from fastapi import FastAPI
from pydantic import BaseModel
from table import RetrievalTable

app = FastAPI()

collection_name = "retrieval-table"
table = RetrievalTable(collection_name=collection_name)

class Query(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str

@app.post("/query/")
async def handle_query(input: Query):
    try:
        response = None
        results = table.query(query=input.query)
        if results:
            response = results["response"]
        return {"query": input.query, "response": response}
    except Exception as e:
        return {'Error has occurred': str(e)}

@app.post("/put/")
async def handle_put(input: QueryResponse):
    try:
        table.put(query=input.query, response=input.response)
        return {"query": input.query, "response": input.response}
    except Exception as e:
        return {'Error has occurred': str(e)}

@app.post("/clear_table/")
async def handle_clear_table():
    try:
        table.clear_table()
        return {"message": "Table cleared"}
    except Exception as e:
        return {'Error has occurred': str(e)}

@app.get("/ping/")
async def ping():
    return {"status": "ok"}

# from fastapi import FastAPI
# from pydantic import BaseModel
# from table import RetrievalTable
# import redis
# import os
# import json
# import time
#
# app = FastAPI()
#
# collection_name = "retrieval-table"
# table = RetrievalTable(collection_name=collection_name)
#
# redis_client = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# DEBATE_QUEUE = "debate_queue"
#
# class Query(BaseModel):
#     query: str
#
# class QueryResponse(BaseModel):
#     query: str
#     response: str
#
# @app.post("/query/")
# async def handle_query(input: Query):
#     try:
#         response = None
#         results = table.query(query=input.query)
#         if results:
#             response = results["response"]
#             return {"query": input.query, "response": response, "source": results.get("match_type", "cache")}
#
#         # No direct or semantic match — fallback to region routing
#         region_ids = get_relevant_parents(input.query)
#
#         if len(region_ids) == 1:
#             # Single parent known — skip debate
#             redis_client.rpush("debate_pool", f"{region_ids[0]}: {input.query}")
#             return {"query": input.query, "response": "Routed directly to parent."}
#
#         elif len(region_ids) > 1:
#             # Push to debate queue with structured metadata
#             debate_payload = {
#                 "query": input.query,
#                 "region_ids": region_ids,
#                 "timestamp": time.time()
#             }
#             redis_client.rpush(DEBATE_QUEUE, json.dumps(debate_payload))
#             return {"query": input.query, "response": "Routed for debate.", "participants": region_ids}
#
#         else:
#             return {"query": input.query, "response": "No relevant parent found."}
#
#     except Exception as e:
#         return {'Error has occurred': str(e)}
#
# @app.post("/put/")
# async def handle_put(input: QueryResponse):
#     try:
#         table.put(query=input.query, response=input.response)
#         return {"query": input.query, "response": input.response}
#     except Exception as e:
#         return {'Error has occurred': str(e)}
#
# @app.post("/clear_table/")
# async def handle_clear_table():
#     try:
#         table.clear_table()
#         return {"message": "Table cleared"}
#     except Exception as e:
#         return {'Error has occurred': str(e)}
#
# @app.get("/ping/")
# async def ping():
#     return {"status": "ok"}
#
# def get_relevant_parents(query):
#     # Placeholder logic — can be replaced with knowledge graph based inference
#     if "France" in query:
#         return ["Parent-EastUS", "Parent-WestUS", "Parent-WestEurope"]
#     elif "Germany" in query:
#         return ["Parent-WestEurope"]
#     return []
