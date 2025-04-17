import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from dataclasses import dataclass, field
import requests
import httpx
import asyncio
from collections import defaultdict
from uuid import uuid4

app = FastAPI()

PARENT_ENDPOINTS = {
    "eastus": "http://74.179.187.11:5001/debate/",
    "westus": "http://13.73.32.15:5001/debate/",
    "westeurope": "http://132.220.120.52:5001/debate/"
}

RETRIEVAL_ID = os.getenv("RETRIEVAL_ID", "retrieval-default")

JUDGE_URL = "http://134.33.161.166:5001/judge/"
TOTAL_ROUNDS = 3

active_debates: Dict[str, Dict] = {}

@dataclass
class DebateRequest():
    query: str
    model: str
    judge_model: str
    total_rounds: int = TOTAL_ROUNDS
    retrieved_knowledge: List[Dict] = field(default_factory=list)

class ParentResponse(BaseModel):
    parent_id: str
    round_number: int
    response: str
    query_id: str


async def post_to_parent(client, parent_id, url, payload):
    try:
        print(f"Contacting {parent_id} for round {payload['round_number']}")
        response = await client.post(url, json=payload, timeout=30)
        print(f"Response from {parent_id}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error contacting {parent_id}: {str(e)}")
        
async def run_debate_rounds(request, query_id, knowledge_dict, PARENT_ENDPOINTS, RETRIEVAL_ID):
    async with httpx.AsyncClient() as client:
        for round_num in range(1, request.total_rounds + 1):
            tasks = []
            for parent_id, url in PARENT_ENDPOINTS.items():
                if parent_id in knowledge_dict:
                    payload = {
                        "query": request.query,
                        "query_id": query_id,
                        "knowledge": knowledge_dict[parent_id],
                        "round_number": round_num,
                        "origin": RETRIEVAL_ID,
                        "model": request.model
                    }
                    tasks.append(post_to_parent(client, parent_id, url, payload))
                else:
                    print(f"Skipping {parent_id} for round {round_num} as not selected.")
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(5)  # Optional small delay before next round
        
async def start_debate(request: DebateRequest):
    try: 
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
        print(knowledge_dict)
        print(f"Starting debate for query: {request.query} with ID: {query_id}")
        await run_debate_rounds(request, query_id, knowledge_dict, PARENT_ENDPOINTS, RETRIEVAL_ID)
            
        all_responses = []
        completed_rounds = active_debates[query_id]["rounds"]
        if completed_rounds:
            last_round_number = max(completed_rounds.keys())
            print(f"Last round number: {last_round_number}")
            all_responses = completed_rounds[last_round_number]
        print(f"All responses collected: {all_responses}")
        
        final_verdict = None
        if all_responses:
            try:
                judge_response = requests.post(JUDGE_URL, json={
                    "query": request.query,
                    "responses": all_responses,
                    "model": request.judge_model,
                })
                final_verdict = judge_response.json()
            except Exception as e:
                print({"query": request.query, "evaluated_responses": all_responses, "response": f"Judge error: {e}"})

        active_debates.pop(query_id, None)
        print(final_verdict)
        if final_verdict is None or "No Answer" in final_verdict.get("response", ""):
            return None
        return final_verdict["response"]
    except Exception as e:
            print(f"Error during debate: {str(e)}")

def submit_response(response: ParentResponse):
    print(f"Received response from parent {response.parent_id} for query {response.query_id} in round {response.round_number}")
    if response.query_id in active_debates:
        round_data = active_debates[response.query_id]["rounds"].setdefault(response.round_number, [])
        round_data.append({
            "parent_id": response.parent_id,
            "response": response.response
        })
        active_debates[response.query_id]["completed_parents"].add(response.parent_id)
    return {"status": "recorded"}

def get_round_responses(round_number: int, query_id: str):
    round_data = active_debates.get(query_id, {}).get("rounds", {}).get(round_number, [])
    print(f"Round {round_number} responses for query {query_id}: {round_data}")
    return {"responses": round_data}
