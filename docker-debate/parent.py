# import os
# import time
# import requests
# from pydantic import BaseModel
# from fastapi import FastAPI
# app = FastAPI()
#
# PARENT_ID = os.getenv("PARENT_ID", "Parent-Default")
# MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma:2b")
# ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:9000")
#
# class DebatePrompt(BaseModel):
#     query: str
#     round_number: int
#
# @app.post("/debate/")
# def participate(prompt: DebatePrompt):
#     # Fetch previous round responses
#     try:
#         res = requests.get(f"{ORCHESTRATOR_URL}/round_responses/{prompt.round_number - 1}")
#         others = res.json().get("responses", [])
#     except:
#         others = []
#
#     context = "\n\n".join([f"{r['parent_id']} said: {r['response']}" for r in others])
#
#     full_prompt = f"""
# You are {PARENT_ID}, participating in a debate.
#
# Query: {prompt.query}
# Round: {prompt.round_number}
#
# Other responses so far:
# {context}
#
# Provide your updated response for this round:
# """
#
#     print(f"[{PARENT_ID}] Sending prompt to model...")
#     res = requests.post("http://localhost:11434/api/generate", json={
#         "model": MODEL_NAME,
#         "prompt": full_prompt,
#         "stream": False
#     })
#     answer = res.json().get("response", "No response")
#
#     requests.post(f"{ORCHESTRATOR_URL}/submit_response", json={
#         "parent_id": PARENT_ID,
#         "round_number": prompt.round_number,
#         "response": answer.strip()
#     })
#
#     return {"status": "submitted"}


import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

PARENT_ID = os.getenv("PARENT_ID", "Parent-Default")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma:2b")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:9000")

class DebatePrompt(BaseModel):
    query: str
    query_id: str
    round_number: int

@app.post("/debate/")
def participate(prompt: DebatePrompt):
    try:
        res = requests.get(f"{ORCHESTRATOR_URL}/round_responses/{prompt.round_number - 1}",
                           params={"query_id": prompt.query_id})
        others = res.json().get("responses", [])
    except:
        others = []

    context = "\n\n".join([f"{r['parent_id']} said: {r['response']}" for r in others])
    full_prompt = f"""
You are {PARENT_ID}, participating in a debate.

Query: {prompt.query}
Round: {prompt.round_number}

Other responses so far:
{context}

Now write your response:
"""

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    })
    answer = res.json().get("response", "No response")

    requests.post(f"{ORCHESTRATOR_URL}/submit_response", json={
        "parent_id": PARENT_ID,
        "round_number": prompt.round_number,
        "response": answer.strip(),
        "query_id": prompt.query_id
    })

    return {"status": "submitted"}
