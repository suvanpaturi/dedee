import redis, os, time, requests

r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

PARENT_ID = os.getenv("PARENT_ID", "Judge-EastUS")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
ROUND_SIZE = 3  # Number of parents

def format_judge_prompt(messages):
    prompt = 'You are a debate judge. Three participants responded to the prompt: "What is the capital of France?"\n\n'
    prompt += '\n'.join(messages)
    prompt += '\n\nPlease evaluate each answer briefly and declare a winner.'
    return prompt

while True:
    try:
        messages = r.lrange("debate_pool", -ROUND_SIZE, -1)
        if len(messages) == ROUND_SIZE:
            prompt = format_judge_prompt(messages)
            print(f"🧾 Judge Prompt:\n{prompt}\n")

            res = requests.post("http://localhost:11434/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            })

            full_response = res.json()
            print("🧠 Raw response from model:", full_response)

            decision = full_response.get("response", "").strip()
            if not decision:
                decision = "No judgement."

            r.rpush("judgement_pool", f"{PARENT_ID}: {decision}")
            print("✅ Judge decision pushed.")
        time.sleep(30)
    except Exception as e:
        print(f"❌ Judge error: {e}")
        time.sleep(15)
