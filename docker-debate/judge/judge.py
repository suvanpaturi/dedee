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
    You are a debate judge tasked with producing the best possible answer to the following question, based on multiple responses.
    
    Question: {data.query}
    
    Responses:
    {formatted_responses}
    
    Instructions:
    - Carefully review all responses.
    - If one response clearly stands out as the best, use it as the final answer.
    - If combining parts of multiple responses leads to a better, more complete answer, synthesize them into one cohesive response.
    - Do not include speaker names, labels, or attribution.
    - Do not add explanations, reasoning, or extra commentary.
    - If the responses are insufficient or contradictory, respond with exactly: No Answer
    
    Final Answer:
    """

    print(f"Final prompt: {final_prompt}")
    try:
        judge_res = requests.post("http://localhost:11434/api/generate", json={
            "model": data.model,
            "prompt": final_prompt,
            "stream": False
        })
        print("Judge response", judge_res.json())
        response = judge_res.json().get("response", "No Answer").strip()
        
        cleanup_prompt = f"""
        You are a post-debate response cleaner. 
        - Removing any filler phrases like "Sure", "Here's what I think", "My final answer is", etc.
        - Removing any redundant or repetitive content.
        - Return only the cleaned answer, nothing else.

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