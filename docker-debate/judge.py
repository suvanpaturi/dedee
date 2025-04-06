# import redis, os, time, requests
#
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# PARENT_ID = os.getenv("PARENT_ID", "Judge-EastUS")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
# ROUND_SIZE = 3  # Number of parents
#
# def format_judge_prompt(messages):
#     prompt = 'You are a debate judge. Three participants responded to the prompt: "What is the capital of France?"\n\n'
#     prompt += '\n'.join(messages)
#     prompt += '\n\nPlease evaluate each answer briefly and declare a winner.'
#     return prompt
#
# while True:
#     try:
#         messages = r.lrange("debate_pool", -ROUND_SIZE, -1)
#         if len(messages) == ROUND_SIZE:
#             prompt = format_judge_prompt(messages)
#             print(f"🧾 Judge Prompt:\n{prompt}\n")
#
#             res = requests.post("http://localhost:11434/api/generate", json={
#                 "model": OLLAMA_MODEL,
#                 "prompt": prompt,
#                 "stream": False
#             })
#
#             full_response = res.json()
#             print("🧠 Raw response from model:", full_response)
#
#             decision = full_response.get("response", "").strip()
#             if not decision:
#                 decision = "No judgement."
#
#             r.rpush("judgement_pool", f"{PARENT_ID}: {decision}")
#             print("✅ Judge decision pushed.")
#         time.sleep(30)
#     except Exception as e:
#         print(f"❌ Judge error: {e}")
#         time.sleep(15)

###########################################################################
# import redis, os, time, json, requests
#
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# PARENT_ID = os.getenv("PARENT_ID", "Judge-EastUS")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
# ROUND_SIZE = 3  # default if not specified in metadata
#
# DEBATE_POOL = "debate_pool"
# JUDGEMENT_POOL = "judgement_pool"
#
# def format_prompt(query, messages):
#     prompt = f'You are a debate judge. Participants responded to: "{query}"\n\n'
#     prompt += "\n".join(messages)
#     prompt += "\n\nPlease evaluate each answer briefly and declare a winner."
#     return prompt
#
# print(f"⚖️ Judge {PARENT_ID} is now listening...")
#
# # while True:
# #     try:
# #         all_items = r.lrange(DEBATE_POOL, 0, -1)
# #         if not all_items:
# #             time.sleep(5)
# #             continue
# #
# #         # Scan backwards to find a recent debate payload
# #         for idx in range(len(all_items)-1, -1, -1):
# #             if not all_items[idx].startswith("{"):
# #                 continue
# #             try:
# #                 meta = json.loads(all_items[idx])
# #                 if not meta.get("region_ids") or not meta.get("query"):
# #                     continue
# #
# #                 region_ids = meta["region_ids"]
# #                 query = meta["query"]
# #                 expected = set(region_ids)
# #
# #                 # Extract the latest matching responses
# #                 responses = []
# #                 for entry in reversed(all_items[idx+1:]):
# #                     for rid in expected:
# #                         if entry.startswith(f"{rid}:"):
# #                             responses.append(entry)
# #                     if len(responses) >= len(expected):
# #                         break
# #
# #                 if len(responses) < len(expected):
# #                     continue  # not all participants responded yet
# #
# #                 full_prompt = format_prompt(query, responses)
# #
# #                 res = requests.post("http://localhost:11434/api/generate", json={
# #                     "model": OLLAMA_MODEL,
# #                     "prompt": full_prompt,
# #                     "stream": False
# #                 })
# #                 decision = res.json().get("response", "No judgement.")
# #                 r.rpush(JUDGEMENT_POOL, f"{PARENT_ID}: {decision}")
# #                 print(f"✅ Judgement pushed for: {query}")
# #                 break
# #
# #             except Exception as e:
# #                 print(f"⚠️ Judge parsing error: {e}")
# #                 continue
# #
# #         time.sleep(10)
# #
# #     except Exception as e:
# #         print(f"❌ Judge error: {e}")
# #         time.sleep(10)
#
# DEBATE_META_POOL = "debate_meta_pool"
#
# while True:
#     try:
#         meta_json = r.lpop(DEBATE_META_POOL)
#         if not meta_json:
#             time.sleep(3)
#             continue
#
#         meta = json.loads(meta_json)
#         region_ids = meta["region_ids"]
#         query = meta["query"]
#         expected = set(region_ids)
#
#         # Wait for responses
#         wait_start = time.time()
#         timeout = 30
#         responses = []
#
#         while time.time() - wait_start < timeout:
#             all_responses = r.lrange(DEBATE_POOL, 0, -1)
#             responses = [r for r in all_responses if query.lower() in r.lower() and any(r.startswith(f"{rid}:") for rid in expected)]
#             if len({r.split(":")[0] for r in responses}) >= len(expected):
#                 break
#             time.sleep(2)
#
#         if len({r.split(":")[0] for r in responses}) < len(expected):
#             print(f"⚠️ Not all participants responded for: {query}")
#             continue
#
#         # Run judge logic
#         prompt = format_prompt(query, responses)
#         res = requests.post("http://localhost:11434/api/generate", json={
#             "model": OLLAMA_MODEL,
#             "prompt": prompt,
#             "stream": False
#         })
#
#         decision = res.json().get("response", "No judgement.")
#         r.rpush(JUDGEMENT_POOL, json.dumps({
#             "query": query,
#             "response": decision,
#             "winning_region": None,
#             "confidence": 0.9
#         }))
#         print(f"✅ Judgement pushed for: {query}")
#
#     except Exception as e:
#         print(f"❌ Judge error: {e}")
#         time.sleep(5)

#############################################################################

# import redis, os, time, json, requests
#
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# PARENT_ID = os.getenv("PARENT_ID", "Judge-EastUS")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
# ROUND_SIZE = 3
#
# DEBATE_POOL = "debate_pool"
# JUDGEMENT_POOL = "judgement_pool"
# DEBATE_META_POOL = "debate_meta_pool"
#
# def format_prompt(query, messages):
#     prompt = f'You are a debate judge. Participants responded to: "{query}"\n'
#     prompt += "\n".join(messages)
#     prompt += "\n\nPlease evaluate each answer briefly and declare a winner."
#     return prompt
#
# print(f"⚖️ Judge {PARENT_ID} is now listening...")
#
# while True:
#     try:
#         meta_json = r.lpop(DEBATE_META_POOL)
#         if not meta_json:
#             time.sleep(3)
#             continue
#
#         meta = json.loads(meta_json)
#         region_ids = meta.get("region_ids", [])
#         query = meta.get("query", "")
#         expected = set(region_ids)
#
#         print(f"🔍 Awaiting responses for: {query} from {expected}")
#
#         wait_start = time.time()
#         timeout = 30
#         responses = []
#
#         while time.time() - wait_start < timeout:
#             all_responses = r.lrange(DEBATE_POOL, 0, -1)
#             matched = [r for r in all_responses if query.lower() in r.lower() and any(r.startswith(f"{rid}:") for rid in expected)]
#             responders = {r.split(":")[0] for r in matched}
#
#             if len(responders) >= len(expected):
#                 responses = matched
#                 break
#             time.sleep(2)
#
#         if len({r.split(":")[0] for r in responses}) < len(expected):
#             print(f"⚠️ Not all participants responded for: {query} — got {[r.split(":")[0] for r in responses]}")
#             continue
#
#         prompt = format_prompt(query, responses)
#         res = requests.post("http://localhost:11434/api/generate", json={
#             "model": OLLAMA_MODEL,
#             "prompt": prompt,
#             "stream": False
#         })
#
#         decision = res.json().get("response", "No judgement.")
#         r.rpush(JUDGEMENT_POOL, json.dumps({
#             "query": query,
#             "response": decision,
#             "winning_region": None,
#             "confidence": 0.9
#         }))
#         print(f"✅ Judgement pushed for: {query}")
#
#     except Exception as e:
#         print(f"❌ Judge error: {e}")
#         time.sleep(5)

###################################NO REDIS JUDGE##########################################
# # judge.py
# import os
# import requests
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
#
# app = FastAPI()
#
# JUDGE_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
#
# class ParentResponse(BaseModel):
#     parent_id: str
#     response: str
#
# class JudgeRequest(BaseModel):
#     query: str
#     responses: List[ParentResponse]
#
# @app.post("/judge/")
# def judge(data: JudgeRequest):
#     prompt = f"You are an impartial judge evaluating responses for the query:\n'{data.query}'\n\n"
#
#     for r in data.responses:
#         prompt += f"{r.parent_id} says: {r.response}\n\n"
#
#     prompt += "Provide a brief evaluation and declare the best response."
#
#     res = requests.post("http://localhost:11434/api/generate", json={
#         "model": JUDGE_MODEL,
#         "prompt": prompt,
#         "stream": False
#     })
#     verdict = res.json().get("response", "").strip()
#
#     return {
#         "query": data.query,
#         "verdict": verdict,
#         "evaluated_responses": data.responses
#     }
######################### WITHOUT REDIS #########################
# import os
# import requests
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
#
# app = FastAPI()
#
# JUDGE_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
#
# class ParentResponse(BaseModel):
#     parent_id: str
#     response: str
#
# class JudgeRequest(BaseModel):
#     query: str
#     responses: List[ParentResponse]
#
#
# def generate_llm_response(prompt: str) -> str:
#     res = requests.post("http://localhost:11434/api/generate", json={
#         "model": JUDGE_MODEL,
#         "prompt": prompt,
#         "stream": False
#     })
#     return res.json().get("response", "").strip()
#
#
# @app.post("/judge/")
# def judge(data: JudgeRequest):
#     formatted_responses = "\n\n".join(
#         [f"{r.parent_id} says: {r.response}" for r in data.responses]
#     )
#
#     final_judge_prompt = f"""
# You are the debate judge. Given multiple responses to the query '{data.query}', carefully evaluate their correctness, depth, and completeness.
#
# Responses:
# {formatted_responses}
#
# Your task is:
# - Summarize the responses into one comprehensive and authoritative final answer.
# - Do NOT list each parent's answer separately.
# - Provide a clear, single-paragraph summary as the definitive answer to the original query.
# """
#
#     final_verdict = generate_llm_response(final_judge_prompt)
#
#     return {
#         "query": data.query,
#         "verdict": final_verdict,
#         "evaluated_responses": data.responses
#     }

########################################################## WORKS -- QUERY REPEATS
#
# import os
# import requests
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
#
# app = FastAPI()
#
# JUDGE_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
#
# class ParentResponse(BaseModel):
#     parent_id: str
#     response: str
#
# class VerdictRequest(BaseModel):
#     query: str
#     responses: List[ParentResponse]
#
# @app.post("/verdict/")
# def evaluate(data: VerdictRequest):
#     formatted = "\n\n".join([f"{r.parent_id} says: {r.response}" for r in data.responses])
#
#     prompt = f"""
# You are the debate judge. Given multiple responses to the query '{data.query}', carefully evaluate their correctness, depth, and completeness.
#
# Responses:
# {formatted}
#
# Your task is:
# - Summarize the responses into one comprehensive and authoritative final answer.
# - Do NOT list each parent's answer separately.
# - Provide a clear, single-paragraph summary as the definitive answer to the original query.
# """
#
#     res = requests.post("http://localhost:11434/api/generate", json={
#         "model": JUDGE_MODEL,
#         "prompt": prompt,
#         "stream": False
#     })
#     verdict = res.json().get("response", "No verdict returned.").strip()
#     return {"verdict": verdict}

import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma:2b")

class ParentResponse(BaseModel):
    parent_id: str
    response: str

class JudgeRequest(BaseModel):
    query: str
    responses: List[ParentResponse]

@app.post("/judge/")
def judge(data: JudgeRequest):
    formatted_responses = "\n\n".join([f"{r.parent_id} says: {r.response}" for r in data.responses])
    final_prompt = f"""
You are the debate judge. Given multiple responses to the query '{data.query}', evaluate their correctness, depth, and completeness.

Responses:
{formatted_responses}

Your task:
- Summarize into one authoritative final answer.
- Do NOT list individual parent answers.
- Write a single-paragraph conclusion.
"""

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": MODEL_NAME,
        "prompt": final_prompt,
        "stream": False
    })

    return {
        "query": data.query,
        "evaluated_responses": data.responses,
        "verdict": res.json().get("response", "").strip()
    }
