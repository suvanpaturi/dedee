import os
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

PARENT_ID = os.getenv("PARENT_ID", "Parent-Default")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma:2b")

class DebatePrompt(BaseModel):
    query: str
    query_id: str
    knowledge: List[Dict]
    round_number: int

@app.post("/debate/")
def participate(prompt: DebatePrompt, request: Request):
    origin = f"{request.client.host}:{request.client.port}"
    try:
        res = requests.get(f"{origin}/round_responses/{prompt.round_number - 1}",
                           params={"query_id": prompt.query_id})
        others = res.json().get("responses", [])
    except:
        others = []

    context = "\n\n".join([f"{r['parent_id']} said: {r['response']}" for r in others])
    knowledge_block = "\n\n".join([
                f"Q: {ex['question']}\nA: {ex['answer']}" for ex in prompt.knowledge
            ])
    
    full_prompt = f"""
You are {PARENT_ID}, participating in a debate.

Query: {prompt.query}
Round: {prompt.round_number}

Other responses so far:
{context}

You have access to the following knowledge consisting of similar question and answers, which could be helpful:
{knowledge_block}

Now write your response:
"""

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    })
    answer = res.json().get("response", "No response")

    requests.post(f"{origin}/submit_response", json={
        "parent_id": PARENT_ID,
        "round_number": prompt.round_number,
        "response": answer.strip(),
        "query_id": prompt.query_id
    })

    return {"status": "submitted"}
