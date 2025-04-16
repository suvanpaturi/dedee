import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import httpx

app = FastAPI()

PARENT_ID = os.getenv("PARENT_ID", "Parent-Default")

class KnowledgeItem(BaseModel):
    question: str
    answer: str
    
class DebatePrompt(BaseModel):
    query: str
    query_id: str
    knowledge: List[KnowledgeItem]
    round_number: int
    origin: str
    model: str

retrieval_agents = {
    "eastus": "128.203.120.105",
    "westus": "52.159.147.210",
    "westeurope": "9.163.200.77"
}

@app.post("/debate/")
async def participate(prompt: DebatePrompt):
    print(prompt)
    print(f"Received prompt from {prompt.origin} retrieval agent: {prompt.query} (Round {prompt.round_number})")
    others = []
    if prompt.round_number != 1:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    f"http://{retrieval_agents[prompt.origin]}:5001/round_responses/{prompt.round_number - 1}",
                    params={"query_id": prompt.query_id}
                )
            others = res.json().get("responses", [])
        except Exception as e:
            print(f"Error fetching previous round responses: {str(e)}")

    context = "\n\n".join([f"{r['parent_id']} said: {r['response']}" for r in others])
    knowledge_block = "\n\n".join([
                f"Q: {ex.question}\nA: {ex.answer}" for ex in prompt.knowledge
            ])
    print(knowledge_block)
    full_prompt = f"""
    You are {PARENT_ID}, participating in a multi-round debate.
    Below is a question, a list of responses from other participants (if any), and a set of relevant Q&A knowledge examples.
    Your task is to generate a clear and informative response to the query, using the provided knowledge when applicable. 
    You may consider the other responses for inspiration or contrast.
    
    Rules:
    - If the answer can be inferred from the knowledge, please provide it confidently.
    - Do not say “I do not know” if the knowledge includes relevant information.
    - Avoid speculation beyond the knowledge unless it is general common sense.
    
    ---

    Question:
    {prompt.query}

    Round: {prompt.round_number}

    ---

    Other responses so far:
    {context}

    ---

    Relevant knowledge examples:
    {knowledge_block}

    ---

    Now write your response:
    """
    print(full_prompt)
    try:
        # Step 1: Generate the response
        ollama_res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": prompt.model,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60
        )
        ollama_res.raise_for_status()
        answer = ollama_res.json().get("response", "No response").strip()

        print("ollama-answer", answer)

        # Step 2: Submit the response
        submit_res = requests.post(
            f"http://{retrieval_agents[prompt.origin]}:5001/submit_response/",
            json={
                "parent_id": PARENT_ID,
                "round_number": prompt.round_number,
                "response": answer,
                "query_id": prompt.query_id
            },
            timeout=10
        )
        submit_res.raise_for_status()
    except Exception as e:
        print(f"Error during response generation or submission: {str(e)}")
        return {"status": "error", "message": str(e)}

    return {"status": "submitted", "answer": answer}