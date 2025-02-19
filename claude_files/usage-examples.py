import requests

def query_ollama(prompt, model="llama2"):
    """
    Send a query to Ollama API
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Simple question
    response = query_ollama("What is the capital of France?")
    print(response['response'])
    
    # Code generation
    code_prompt = "Write a Python function to calculate fibonacci numbers"
    response = query_ollama(code_prompt)
    print(response['response'])
