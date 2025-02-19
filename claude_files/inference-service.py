from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import os
from typing import Optional, Dict, List

app = FastAPI(title="Ollama Inference Service")

class InferenceRequest(BaseModel):
    prompt: str
    model: str = "llama2"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    context_length: Optional[int] = 4096
    max_tokens: Optional[int] = 1000

class ModelConfig(BaseModel):
    name: str
    parameters: Dict
    is_available: bool = False

# Model configurations
MODEL_CONFIGS = {
    "llama2": {
        "name": "llama2",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "context_length": 4096,
            "num_gpu": 1 if os.getenv("NVIDIA_VISIBLE_DEVICES") else 0
        }
    },
    "mistral": {
        "name": "mistral",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "context_length": 4096,
            "num_gpu": 1 if os.getenv("NVIDIA_VISIBLE_DEVICES") else 0
        }
    },
    "orca-mini": {
        "name": "orca-mini",
        "parameters": {
            "temperature": 0.7,
            "top_p": 0.9,
            "context_length": 2048,
            "num_gpu": 1 if os.getenv("NVIDIA_VISIBLE_DEVICES") else 0
        }
    }
}

class InferenceService:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.available_models = self._check_available_models()

    def _check_available_models(self) -> List[str]:
        """Check which models are available in the local storage"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []

    async def generate(self, request: InferenceRequest) -> Dict:
        """Generate response using specified model"""
        if request.model not in self.available_models:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

        try:
            # Prepare request payload
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                }
            }

            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return {
                    "model": request.model,
                    "response": response.json()['response'],
                    "parameters_used": payload["options"]
                }
            else:
                raise HTTPException(status_code=500, detail="Inference failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app and inference service
inference_service = InferenceService()

@app.get("/models")
async def list_models():
    """List all available models and their configurations"""
    available_models = {}
    for model_name, config in MODEL_CONFIGS.items():
        config["is_available"] = model_name in inference_service.available_models
        available_models[model_name] = config
    return available_models

@app.post("/generate")
async def generate(request: InferenceRequest):
    """Generate text using specified model"""
    return await inference_service.generate(request)

@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "available_models": inference_service.available_models
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
