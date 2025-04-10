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
    print(f"Received responses for query: {data.query} from parent")
    print(f"Responses: {data.responses}")
    formatted_responses = "\n\n".join([f"{r.parent_id} says: {r.response}" for r in data.responses])
    final_prompt = f"""
You are the debate judge. Given multiple responses to the query '{data.query}', evaluate their correctness, depth, and completeness.

Responses:
{formatted_responses}

Your task:
- Summarize into one succint authoritative final answer.
- Do NOT list individual parent answers.
- Do NOT include any additional information or explanations.
- If final answer is not clear, just state "No Answer" nothing else.
"""
    print(f"Final prompt: {final_prompt}")
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": MODEL_NAME,
            "prompt": final_prompt,
            "stream": False
        })
        print(res.text)
        return {
            "query": data.query,
            "evaluated_responses": data.responses,
            "response": res.json().get("response", "").strip()
        }
    except Exception as e:
        print(f"Error during response generation or submission: {str(e)}")
        return {"status": "error", "message": str(e)}