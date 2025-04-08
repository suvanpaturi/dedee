import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import requests
from collections import defaultdict
from uuid import uuid4

app = FastAPI()

PARENT_ENDPOINTS = {
    "east-us": "http://ip:5001/debate/",
    "west-us": "http://ip:localhost:5001/debate/",
    "west-europe": "http://ip:localhost:5001/debate/"
}

JUDGE_URL = "http://ip:5001/judge/"
TOTAL_ROUNDS = 3

active_debates: Dict[str, Dict] = {}

class DebateRequest():
    query: str
    total_rounds: int = TOTAL_ROUNDS
    retrieved_knowledge: List[Dict] = []

class ParentResponse(BaseModel):
    parent_id: str
    round_number: int
    response: str
    query_id: str

def start_debate(request: DebateRequest):

    query_id = str(uuid4())
    active_debates[query_id] = {
        "query": request.query,
        "rounds": {},
        "completed_parents": set()
    }
    
    knowledge_dict = defaultdict(list)
    for item in request.retrieved_knowledge:
        knowledge_dict[item["region_id"]].append({ #since each parent has a different region_id
            "question": item["query_text"],
            "answer": item["answer_text"]
    })
        
    for round_num in range(1, request.total_rounds + 1):
        for parent_id, url in PARENT_ENDPOINTS.items():
            if parent_id in knowledge_dict:
                try:
                    requests.post(url, json={
                        "query": request.query,
                        "query_id": query_id,
                        "knowledge": knowledge_dict[parent_id],
                        "round_number": round_num
                    })
                except Exception as e:
                    print(f"Error contacting {parent_id}: {e}")
            else:
                print(f"Skipping {parent_id} for round {round_num} as not selected.")
        time.sleep(3)  # Give parents time to respond

    all_responses = []
    for round_data in active_debates[query_id]["rounds"].values():
        all_responses.extend(round_data)

    try:
        judge_response = requests.post(JUDGE_URL, json={
            "query": request.query,
            "responses": all_responses
        })
        final_verdict = judge_response.json()
    except Exception as e:
        final_verdict = {"query": request.query, "evaluated_responses": all_responses, "response": f"Judge error: {e}"}

    del active_debates[query_id]
    print(final_verdict)
    return final_verdict["response"] if final_verdict["response"] != "No Answer" else None

@app.post("/submit_response")
def submit_response(response: ParentResponse):
    if response.query_id in active_debates:
        round_data = active_debates[response.query_id]["rounds"].setdefault(response.round_number, [])
        round_data.append({
            "parent_id": response.parent_id,
            "response": response.response
        })
        active_debates[response.query_id]["completed_parents"].add(response.parent_id)
    return {"status": "recorded"}

@app.get("/round_responses/{round_number}")
def get_round_responses(round_number: int, query_id: str):
    round_data = active_debates.get(query_id, {}).get("rounds", {}).get(round_number, [])
    return {"responses": round_data}
