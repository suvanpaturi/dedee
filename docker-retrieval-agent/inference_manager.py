# from fastapi import FastAPI
# from pydantic import BaseModel
# import redis
# import os
# import json
# import time
# import asyncio
# from retrieval_table.table import RetrievalTable
#
# app = FastAPI()
#
# # Redis connection
# redis_client = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# collection_name = "retrieval-table"
# table = RetrievalTable(collection_name=collection_name)
# DEBATE_QUEUE = "debate_queue"
#
# class Query(BaseModel):
#     query: str
#
# def get_relevant_parents(query: str):
#     # Placeholder for KG integration — refine later
#     if "France" in query:
#         return ["Parent-EastUS", "Parent-WestUS", "Parent-WestEurope"]
#     elif "Germany" in query:
#         return ["Parent-WestEurope"]
#     return []
#
# @app.post("/inference/")
# async def handle_inference(input: Query):
#     try:
#         # 1. Check retrieval table first
#         results = table.query(query=input.query)
#         if results:
#             return {
#                 "query": input.query,
#                 "response": results["response"],
#                 "source": results.get("match_type", "cache")
#             }
#
#         # 2. Determine relevant parents (from retrieval or KG)
#         region_ids = get_relevant_parents(input.query)
#
#         if len(region_ids) == 1:
#             redis_client.rpush("debate_pool", f"{region_ids[0]}: {input.query}")
#             return {
#                 "query": input.query,
#                 "response": "Sent directly to region",
#                 "region": region_ids[0]
#             }
#
#         elif len(region_ids) > 1:
#             debate_payload = {
#                 "query": input.query,
#                 "region_ids": region_ids,
#                 "timestamp": time.time()
#             }
#             redis_client.rpush(DEBATE_QUEUE, json.dumps(debate_payload))
#
#             # ⏳ Poll judgement_pool for response
#             timeout = 30
#             start = time.time()
#             while time.time() - start < timeout:
#                 judgements = redis_client.lrange("judgement_pool", -20, -1)
#                 for j in reversed(judgements):
#                     if input.query.lower() in j.lower():
#                         return {
#                             "query": input.query,
#                             "response": j,
#                             "source": "debate_result"
#                         }
#                 await asyncio.sleep(2)
#
#             return {
#                 "query": input.query,
#                 "response": "✅ Debate initiated but no judgement returned yet.",
#                 "participants": region_ids
#             }
#
#         else:
#             return {
#                 "query": input.query,
#                 "response": "No relevant parents found"
#             }
#
#     except Exception as e:
#         return {"error": str(e)}
#
# @app.get("/result/")
# async def get_final_judgement(query: str):
#     try:
#         judgements = redis_client.lrange("judgement_pool", -50, -1)
#         for j in reversed(judgements):
#             if query.lower() in j.lower():
#                 return {
#                     "query": query,
#                     "judgement": j
#                 }
#         return {
#             "query": query,
#             "judgement": "❌ No matching judgement found yet."
#         }
#     except Exception as e:
#         return {"error": str(e)}
#
# # @app.get("/responses/")
# # async def get_parent_responses(query: str):
# #     try:
# #         all_items = redis_client.lrange("debate_pool", -1000, -1)  # check only recent ones
# #         responses = []
# #
# #         for item in reversed(all_items):
# #             if item.startswith("{"):
# #                 try:
# #                     meta = json.loads(item)
# #                     if meta.get("query", "").lower() == query.lower():
# #                         break  # found the debate metadata block, stop there
# #                 except:
# #                     continue
# #             elif query.lower() in item.lower():
# #                 responses.append(item)
# #
# #         if responses:
# #             return {
# #                 "query": query,
# #                 "parent_responses": responses
# #             }
# #         return {
# #             "query": query,
# #             "parent_responses": "❌ No parent responses found (even loosely matched)"
# #         }
# #
# #     except Exception as e:
# #         return {"error": str(e)}

from fastapi import FastAPI
from pydantic import BaseModel
import redis
import os
import json
import time
import asyncio
from retrieval_table.table import RetrievalTable

app = FastAPI()

# Redis connection
redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

collection_name = "retrieval-table"
table = RetrievalTable(collection_name=collection_name)
DEBATE_QUEUE = "debate_queue"

class Query(BaseModel):
    query: str

def get_relevant_parents(query: str):
    # Placeholder for KG integration — refine later
    if "France" in query:
        return ["Parent-EastUS", "Parent-WestUS", "Parent-WestEurope"]
    elif "Germany" in query:
        return ["Parent-WestEurope"]
    return []

@app.post("/inference/")
async def handle_inference(input: Query):
    try:
        # 1. Check retrieval table first
        results = table.query(query=input.query)
        if results:
            return {
                "query": input.query,
                "response": results["response"],
                "source": results.get("match_type", "cache")
            }

        # 2. Determine relevant parents (from retrieval or KG)
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

            # ⏳ Poll judgement_pool for response
            timeout = 30
            start = time.time()
            while time.time() - start < timeout:
                judgements = redis_client.lrange("judgement_pool", -20, -1)
                for j in reversed(judgements):
                    if input.query.lower() in j.lower():
                        # ✅ Store judged result in table
                        table.put(input.query, response=j, region="Judged")

                        return {
                            "query": input.query,
                            "response": j,
                            "source": "debate_result"
                        }
                await asyncio.sleep(2)

            return {
                "query": input.query,
                "response": "✅ Debate initiated but no judgement returned yet.",
                "participants": region_ids
            }

        else:
            return {
                "query": input.query,
                "response": "No relevant parents found"
            }

    except Exception as e:
        return {"error": str(e)}
