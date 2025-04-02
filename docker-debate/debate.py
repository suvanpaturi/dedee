import redis
import os
import time
import requests

r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
MODEL_NAME = os.getenv("OLLAMA_MODEL")  # e.g., "gemma:2b", "tinyllama", "qwen:0.5b"

while True:
    prompt = "What is the capital of France?"
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
        )
        answer = res.json().get("response", "No response")
        r.rpush("debate_pool", f"{PARENT_ID}: {answer}")
    except Exception as e:
        r.rpush("debate_pool", f"{PARENT_ID}: ERROR - {e}")

    time.sleep(10)
