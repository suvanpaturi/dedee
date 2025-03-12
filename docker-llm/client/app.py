from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .llm_client import OllamaClient

app = FastAPI()

llm_client = OllamaClient()

class GenerateRequest(BaseModel):
    model: str
    prompt: str

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Endpoint to generate text based on a given prompt.
    """
    try:
        response = llm_client.generate(model=request.model, prompt=request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))