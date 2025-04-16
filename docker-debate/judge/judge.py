import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ParentResponse(BaseModel):
    parent_id: str
    response: str

class JudgeRequest(BaseModel):
    query: str
    responses: List[ParentResponse]
    model: str

@app.post("/judge/")
def judge(data: JudgeRequest):
    print(f"Received responses for query: {data.query} from parent")
    print(f"Responses: {data.responses}")
    formatted_responses = "\n\n".join([f"{r.parent_id} says: {r.response}" for r in data.responses])
    final_prompt = f"""
    You are the debate judge. You are given multiple responses to the query '{data.query}'.
    Based on these decide, decide on a final answer.

    Responses:
    {formatted_responses}

    Your task:
    - Do NOT include any additional information or explanations.
    - If final answer can't be decided from responses, just state "No Answer" nothing else.
    """

    print(f"Final prompt: {final_prompt}")
    try:
        judge_res = requests.post("http://localhost:11434/api/generate", json={
            "model": data.model,
            "prompt": final_prompt,
            "stream": False
        })
        response = judge_res.json().get("response", "No Answer").strip()
        
        cleanup_prompt = f"""
        You are a post-debate response cleaner. Your job is to return a **succinct, standalone final answer** to a question or prompt.

        Clean the response by:
        - Removing any filler phrases like "Sure", "Here's what I think", "My final answer is", etc.
        - Removing any references to a debate, previous discussion, or multiple viewpoints.
        - Removing any redundant or repetitive content.
        - Only include the **final answer** in a clear, complete sentence.
        - If the answer is not clear, respond with: "No Answer".

        Return only the cleaned answer, nothing else.

        Response to clean:
        {response}
        """
        
        print(f"Cleanup prompt: {cleanup_prompt}")
        final_res = requests.post("http://localhost:11434/api/generate", json={
            "model": data.model,
            "prompt": cleanup_prompt,
            "stream": False
        })
        final_response = final_res.json().get("response", "No Answer").strip()
        print(f"Final response: {final_response}")
        return {
            "query": data.query,
            "evaluated_responses": data.responses,
            "response": final_response
        }
    except Exception as e:
        print(f"Error during response generation or submission: {str(e)}")
        return {"status": "error", "message": str(e)}