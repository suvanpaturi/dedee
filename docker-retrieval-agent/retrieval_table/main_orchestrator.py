# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List, Dict
# import requests
# import uvicorn
#
# app = FastAPI()
#
# # Store all round responses in memory
# DEBATE_STATE: Dict[int, List[Dict]] = {}
# PARENT_ENDPOINTS = [
#     ("Parent-EastUS", "http://localhost:9101"),
#     ("Parent-WestUS", "http://localhost:9102"),
#     ("Parent-Europe", "http://localhost:9103")
# ]
# JUDGE_URL = "http://localhost:9200/verdict/"
#
# class QueryRequest(BaseModel):
#     query: str
#     total_rounds: int = 2
#
# class ParentSubmission(BaseModel):
#     parent_id: str
#     round_number: int
#     response: str
#
# @app.post("/start_debate/")
# def start_debate(q: QueryRequest):
#     for round_number in range(1, q.total_rounds + 1):
#         for _, endpoint in PARENT_ENDPOINTS:
#             requests.post(f"{endpoint}/debate", json={
#                 "query": q.query,
#                 "round_number": round_number
#             })
#
#         # Wait for responses (simplified sync)
#         import time
#         while True:
#             responses = DEBATE_STATE.get(round_number, [])
#             if len(responses) >= len(PARENT_ENDPOINTS):
#                 break
#             time.sleep(2)
#
#     # Send final verdict request
#     final_round = q.total_rounds
#     all_final_responses = DEBATE_STATE.get(final_round, [])
#     res = requests.post(JUDGE_URL, json={
#         "query": q.query,
#         "responses": all_final_responses
#     })
#     return {
#         "query": q.query,
#         "evaluated_responses": all_final_responses,
#         "verdict": res.json().get("verdict")
#     }
#
# @app.get("/round_responses/{round_number}")
# def get_round_responses(round_number: int):
#     return {"responses": DEBATE_STATE.get(round_number, [])}
#
# @app.post("/submit_response")
# def receive_response(resp: ParentSubmission):
#     if resp.round_number not in DEBATE_STATE:
#         DEBATE_STATE[resp.round_number] = []
#     DEBATE_STATE[resp.round_number].append({
#         "parent_id": resp.parent_id,
#         "response": resp.response
#     })
#     return {"status": "stored"}
#
# # Run: uvicorn main_orchestrator:app --port 9000
#
# # Each parent runs on its own port: 9101, 9102, 9103
# # Judge runs on 9200

# orchestrator.py
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import requests
from uuid import uuid4

app = FastAPI()

PARENT_ENDPOINTS = {
    "Parent-EastUS": "http://localhost:9101/debate/",
    "Parent-WestUS": "http://localhost:9102/debate/",
    "Parent-Europe": "http://localhost:9103/debate/"
}

JUDGE_URL = "http://localhost:9200/judge/"
TOTAL_ROUNDS = 2
retrieval_table = {}  # In-memory cache
active_debates: Dict[str, Dict] = {}

class DebateRequest(BaseModel):
    query: str
    total_rounds: int = TOTAL_ROUNDS

class ParentResponse(BaseModel):
    parent_id: str
    round_number: int
    response: str
    query_id: str

@app.post("/start_debate/")
def start_debate(request: DebateRequest):
    if request.query in retrieval_table:
        return retrieval_table[request.query]

    query_id = str(uuid4())
    active_debates[query_id] = {
        "query": request.query,
        "rounds": {},
        "completed_parents": set()
    }

    for round_num in range(1, request.total_rounds + 1):
        for parent_id, url in PARENT_ENDPOINTS.items():
            try:
                requests.post(url, json={
                    "query": request.query,
                    "query_id": query_id,
                    "round_number": round_num
                })
            except Exception as e:
                print(f"Error contacting {parent_id}: {e}")
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
        final_verdict = {"query": request.query, "verdict": f"Judge error: {e}", "evaluated_responses": all_responses}

    retrieval_table[request.query] = final_verdict
    del active_debates[query_id]

    return final_verdict

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
