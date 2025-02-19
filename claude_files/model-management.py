import json
import os
import time
import requests
import yaml
from typing import Dict, List

class OllamaManager:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.models_config = self.load_models_config()
        self.models_to_load = os.getenv('MODELS', '').split(',')
        self.default_model = os.getenv('DEFAULT_MODEL', 'llama2')

    def load_models_config(self) -> Dict:
        """Load model configurations from JSON file"""
        with open('/models.json', 'r') as f:
            return json.load(f)['models']

    def pull_model(self, model_name: str) -> bool:
        """Pull a specific model"""
        try:
            config = self.models_config.get(model_name, {}).get('config', {})
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "config": config}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error pulling model {model_name}: {str(e)}")
            return False

    def configure_model(self, model_name: str) -> bool:
        """Configure a specific model"""
        try:
            config = self.models_config.get(model_name, {}).get('config', {})
            response = requests.post(
                f"{self.base_url}/api/create",
                json={"name": model_name, "config": config}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error configuring model {model_name}: {str(e)}")
            return False

    def wait_for_ollama(self, timeout: int = 60):
        """Wait for Ollama service to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    return True
            except:
                time.sleep(1)
        return False

    def initialize_models(self):
        """Initialize all specified models"""
        if not self.wait_for_ollama():
            raise Exception("Ollama service not available")

        for model in self.models_to_load:
            print(f"Initializing model: {model}")
            if self.pull_model(model):
                if self.configure_model(model):
                    print(f"Successfully initialized {model}")
                else:
                    print(f"Failed to configure {model}")
            else:
                print(f"Failed to pull {model}")

    def get_model_status(self) -> List[Dict]:
        """Get status of all models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.json().get('models', [])
        except Exception as e:
            print(f"Error getting model status: {str(e)}")
            return []

if __name__ == "__main__":
    manager = OllamaManager()
    manager.initialize_models()
