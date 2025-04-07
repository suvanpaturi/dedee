# import redis, os, time, requests
#
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
# MODEL_NAME = os.getenv("OLLAMA_MODEL")
# HISTORY_LENGTH = 6
#
# def get_recent_context():
#     # Fetch last N responses
#     recent_responses = r.lrange("debate_pool", -HISTORY_LENGTH, -1)
#     recent_judgements = r.lrange("judgement_pool", -2, -1)
#     context = "\n".join(recent_responses + recent_judgements)
#     return context
#
# while True:
#     try:
#         context = get_recent_context()
#         prompt = f"""You are participating in an ongoing debate. The topic is: "What is the capital of France?"
#
# Here are recent responses and judge decisions:
# {context}
#
# Now, respond thoughtfully, using new information if possible."""
#
#         res = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
#         )
#         answer = res.json().get("response", "No response")
#         r.rpush("debate_pool", f"{PARENT_ID}: {answer}")
#         print("🧠 Pushed contextual response.")
#     except Exception as e:
#         r.rpush("debate_pool", f"{PARENT_ID}: ERROR - {e}")
#         print(f"❌ Debate error: {e}")
#
#     time.sleep(10)
##################################################################################
# import redis, os, time, json, requests
#
# # Redis connection
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# DEBATE_QUEUE = "debate_queue"
# DEBATE_POOL = "debate_pool"
#
# PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
# MODEL_NAME = os.getenv("OLLAMA_MODEL")
#
# print(f"🤖 Starting parent {PARENT_ID}...")
#
# def generate_response(prompt):
#     try:
#         res = requests.post("http://localhost:11434/api/generate", json={
#             "model": MODEL_NAME,
#             "prompt": prompt,
#             "stream": False
#         })
#         return res.json().get("response", "No response")
#     except Exception as e:
#         return f"ERROR - {e}"
#
# while True:
#     try:
#         # Non-blocking pop from left of debate queue
#         payload_json = r.lpop(DEBATE_QUEUE)
#         if not payload_json:
#             time.sleep(2)
#             continue
#
#         payload = json.loads(payload_json)
#         query = payload.get("query", "")
#         region_ids = payload.get("region_ids", [])
#
#         # if PARENT_ID not in region_ids:
#         #     print(f"⏭ {PARENT_ID} skipped debate for: {query}")
#         #     continue  # not for this parent
#         if not any(region.lower() in PARENT_ID.lower() for region in region_ids):
#             print(f"⏭ {PARENT_ID} skipped debate for: {query}")
#             continue
#
#         full_prompt = f"""You are {PARENT_ID}, participating in a debate.
#
# The prompt is: "{query}"
#
# Provide your best possible answer using your knowledge and reasoning.
# """
#         answer = generate_response(full_prompt)
#         r.rpush(DEBATE_POOL, f"{PARENT_ID}: {answer}")
#         print(f"✅ {PARENT_ID} responded to: {query}")
#
#     except Exception as e:
#         r.rpush(DEBATE_POOL, f"{PARENT_ID}: ERROR - {e}")
#         print(f"❌ {PARENT_ID} error: {e}")
#
#     time.sleep(5)

############################################################################

# # ✅ Updated `debate.py`
# import redis, os, time, json, requests
#
# # Redis connection
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# DEBATE_QUEUE = "debate_queue"
# DEBATE_POOL = "debate_pool"
#
# PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
# MODEL_NAME = os.getenv("OLLAMA_MODEL")
#
# print(f"\U0001F916 Starting parent {PARENT_ID}...")
#
# def generate_response(prompt):
#     try:
#         res = requests.post("http://localhost:11434/api/generate", json={
#             "model": MODEL_NAME,
#             "prompt": prompt,
#             "stream": False
#         })
#         print(f"🔁 [{PARENT_ID}] Model raw response: {res.status_code}")
#         return res.json().get("response", "No response")
#     except Exception as e:
#         print(f"❌ [{PARENT_ID}] Error generating response: {e}")
#         return f"ERROR - {e}"
#
# while True:
#     try:
#         payload_json = r.lpop(DEBATE_QUEUE)
#         print(f"📥 [{PARENT_ID}] polled: {payload_json}")
#
#         if not payload_json:
#             time.sleep(2)
#             continue
#
#         payload = json.loads(payload_json)
#         query = payload.get("query", "")
#         region_ids = payload.get("region_ids", [])
#
#         if not any(region.lower() in PARENT_ID.lower() for region in region_ids):
#             print(f"⏭ {PARENT_ID} skipped debate for: {query} | expected: {region_ids}")
#             continue
#
#         full_prompt = f"""You are {PARENT_ID}, participating in a debate.
#
# The prompt is: \"{query}\"
#
# Provide your best possible answer using your knowledge and reasoning.
# """
#         answer = generate_response(full_prompt)
#         # r.rpush(DEBATE_POOL, f"{PARENT_ID}: {answer}")
#         r.rpush(DEBATE_POOL, f"{PARENT_ID}: {query} ||| {answer}")
#         print(f"✅ {PARENT_ID} responded to: {query}")
#
#     except Exception as e:
#         print(f"❌ {PARENT_ID} main loop error: {e}")
#         r.rpush(DEBATE_POOL, f"{PARENT_ID}: ERROR - {e}")
#
#     time.sleep(5)
############################################################################
# import redis, os, time, json, requests
#
# # Redis setup
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
# r.delete("debate_queue", "debate_pool", "debate_meta_pool", "judgement_pool")
#
# # if os.getenv("CLEAR_REDIS_ON_START", "false").lower() == "true":
# #     print(f"🧹 [{PARENT_ID}] Clearing Redis debate keys...")
# #     r.delete("debate_queue", "debate_pool", "debate_meta_pool", "judgement_pool")
#
#
# PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
# MODEL_NAME = os.getenv("OLLAMA_MODEL")
#
# DEBATE_QUEUE = "debate_queue"
# DEBATE_POOL = "debate_pool"
#
# print(f"🤖 Starting parent {PARENT_ID}...")
#
# def generate_response(prompt):
#     try:
#         res = requests.post("http://localhost:11434/api/generate", json={
#             "model": MODEL_NAME,
#             "prompt": prompt,
#             "stream": False
#         })
#         print(f"🔁 [{PARENT_ID}] Model raw response: {res.status_code}")
#         return res.json().get("response", "No response")
#     except Exception as e:
#         print(f"❌ [{PARENT_ID}] Error generating response: {e}")
#         return f"ERROR - {e}"
#
# while True:
#     try:
#         payload_json = r.lpop(DEBATE_QUEUE)
#         print(f"📥 [{PARENT_ID}] polled: {payload_json}")
#
#         if not payload_json:
#             time.sleep(2)
#             continue
#
#         payload = json.loads(payload_json)
#         query = payload.get("query", "")
#         region_ids = payload.get("region_ids", [])
#
#         if not any(region.lower() in PARENT_ID.lower() for region in region_ids):
#             print(f"⏭ {PARENT_ID} skipped debate for: {query} | expected: {region_ids}")
#             continue
#
#         print(f"🎤 {PARENT_ID} joining debate on: {query}")
#         full_prompt = f"""You are {PARENT_ID}, participating in a debate.
#
# The prompt is: \"{query}\"
#
# Provide your best possible answer using your knowledge and reasoning.
# """
#         answer = generate_response(full_prompt)
#         region_id = PARENT_ID.split("-")[-1].lower()  # eastus, westus, westeurope
#         r.rpush(DEBATE_POOL, f"{region_id}: {query} ||| {answer}")
#         # r.rpush(DEBATE_POOL, f"{PARENT_ID}: {query} ||| {answer}")
#         print(f"✅ {PARENT_ID} responded to: {query}")
#
#     except Exception as e:
#         err_msg = f"{PARENT_ID}: {query} ||| ERROR - {e}"
#         r.rpush(DEBATE_POOL, err_msg)
#         print(f"❌ {err_msg}")
#
#     time.sleep(5)

################################NO REDIS DEBATE ############################################
# debate.py
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma:2b")
PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")


class Query(BaseModel):
    query: str

@app.post("/respond/")
def respond(q: Query):
    prompt = f"You are {PARENT_ID}. Answer the question concisely and clearly:\n{q.query}"

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    response = res.json().get("response", "").strip()

    return {"parent_id": PARENT_ID, "response": response}
