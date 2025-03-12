import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

class OllamaClient:
    def __init__(self, base_url=OLLAMA_API_URL):
        self.base_url = base_url

    def chat(self, model: str, prompt: str):
        data = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(self.base_url, json=data)

        if response.status_code == 200:
            return response.json().get("response", "No response")
        return f"Error: {response.status_code} - {response.text}"

    def generate(self, model: str, prompt: str):
        return self.chat(model, prompt)