import requests
import json

# Change if running on a different host/port
API_URL = "http://localhost:8000/inference/"

# Sample queries to test
test_queries = [
    "Tell me about the capital of France",
    "Explain Germany's political structure",
    "What was the weather like in France during the revolution?",
    "Query with no expected region"
]

for q in test_queries:
    print(f"\n🧪 Sending Query: {q}")
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": q})
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Response:", json.dumps(data, indent=2))
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Request failed: {e}")
