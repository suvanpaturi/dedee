# import redis, os, time, requests
#
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
# MODEL_NAME = os.getenv("OLLAMA_MODEL")
# HISTORY_LENGTH = 6
#
# def get_recent_context():
#     # Fetch last N responses
#     recent_responses = r.lrange("debate_pool", -HISTORY_LENGTH, -1)
#     recent_judgements = r.lrange("judgement_pool", -2, -1)
#     context = "\n".join(recent_responses + recent_judgements)
#     return context
#
# while True:
#     try:
#         context = get_recent_context()
#         prompt = f"""You are participating in an ongoing debate. The topic is: "What is the capital of France?"
#
# Here are recent responses and judge decisions:
# {context}
#
# Now, respond thoughtfully, using new information if possible."""
#
#         res = requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
#         )
#         answer = res.json().get("response", "No response")
#         r.rpush("debate_pool", f"{PARENT_ID}: {answer}")
#         print("🧠 Pushed contextual response.")
#     except Exception as e:
#         r.rpush("debate_pool", f"{PARENT_ID}: ERROR - {e}")
#         print(f"❌ Debate error: {e}")
#
#     time.sleep(10)

import redis, os, time, json, requests

# Redis connection
r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

DEBATE_QUEUE = "debate_queue"
DEBATE_POOL = "debate_pool"

PARENT_ID = os.getenv("PARENT_ID", "Parent-Unknown")
MODEL_NAME = os.getenv("OLLAMA_MODEL")

print(f"🤖 Starting parent {PARENT_ID}...")

def generate_response(prompt):
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", "No response")
    except Exception as e:
        return f"ERROR - {e}"

while True:
    try:
        # Non-blocking pop from left of debate queue
        payload_json = r.lpop(DEBATE_QUEUE)
        if not payload_json:
            time.sleep(2)
            continue

        payload = json.loads(payload_json)
        query = payload.get("query", "")
        region_ids = payload.get("region_ids", [])

        if PARENT_ID not in region_ids:
            print(f"⏭ {PARENT_ID} skipped debate for: {query}")
            continue  # not for this parent

        full_prompt = f"""You are {PARENT_ID}, participating in a debate.

The prompt is: "{query}"

Provide your best possible answer using your knowledge and reasoning.
"""
        answer = generate_response(full_prompt)
        r.rpush(DEBATE_POOL, f"{PARENT_ID}: {answer}")
        print(f"✅ {PARENT_ID} responded to: {query}")

    except Exception as e:
        r.rpush(DEBATE_POOL, f"{PARENT_ID}: ERROR - {e}")
        print(f"❌ {PARENT_ID} error: {e}")

    time.sleep(5)
