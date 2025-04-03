# import redis, os, time, requests
#
# r = redis.StrictRedis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT", 6379)),
#     password=os.getenv("REDIS_PASSWORD"),
#     decode_responses=True
# )
#
# PARENT_ID = os.getenv("PARENT_ID", "Judge-EastUS")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
# ROUND_SIZE = 3  # Number of parents
#
# def format_judge_prompt(messages):
#     prompt = 'You are a debate judge. Three participants responded to the prompt: "What is the capital of France?"\n\n'
#     prompt += '\n'.join(messages)
#     prompt += '\n\nPlease evaluate each answer briefly and declare a winner.'
#     return prompt
#
# while True:
#     try:
#         messages = r.lrange("debate_pool", -ROUND_SIZE, -1)
#         if len(messages) == ROUND_SIZE:
#             prompt = format_judge_prompt(messages)
#             print(f"🧾 Judge Prompt:\n{prompt}\n")
#
#             res = requests.post("http://localhost:11434/api/generate", json={
#                 "model": OLLAMA_MODEL,
#                 "prompt": prompt,
#                 "stream": False
#             })
#
#             full_response = res.json()
#             print("🧠 Raw response from model:", full_response)
#
#             decision = full_response.get("response", "").strip()
#             if not decision:
#                 decision = "No judgement."
#
#             r.rpush("judgement_pool", f"{PARENT_ID}: {decision}")
#             print("✅ Judge decision pushed.")
#         time.sleep(30)
#     except Exception as e:
#         print(f"❌ Judge error: {e}")
#         time.sleep(15)
import redis, os, time, json, requests

r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

PARENT_ID = os.getenv("PARENT_ID", "Judge-EastUS")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
ROUND_SIZE = 3  # default if not specified in metadata

DEBATE_POOL = "debate_pool"
JUDGEMENT_POOL = "judgement_pool"

def format_prompt(query, messages):
    prompt = f'You are a debate judge. Participants responded to: "{query}"\n\n'
    prompt += "\n".join(messages)
    prompt += "\n\nPlease evaluate each answer briefly and declare a winner."
    return prompt

print(f"⚖️ Judge {PARENT_ID} is now listening...")

while True:
    try:
        all_items = r.lrange(DEBATE_POOL, 0, -1)
        if not all_items:
            time.sleep(5)
            continue

        # Scan backwards to find a recent debate payload
        for idx in range(len(all_items)-1, -1, -1):
            if not all_items[idx].startswith("{"):
                continue
            try:
                meta = json.loads(all_items[idx])
                if not meta.get("region_ids") or not meta.get("query"):
                    continue

                region_ids = meta["region_ids"]
                query = meta["query"]
                expected = set(region_ids)

                # Extract the latest matching responses
                responses = []
                for entry in reversed(all_items[idx+1:]):
                    for rid in expected:
                        if entry.startswith(f"{rid}:"):
                            responses.append(entry)
                    if len(responses) >= len(expected):
                        break

                if len(responses) < len(expected):
                    continue  # not all participants responded yet

                full_prompt = format_prompt(query, responses)

                res = requests.post("http://localhost:11434/api/generate", json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False
                })
                decision = res.json().get("response", "No judgement.")
                r.rpush(JUDGEMENT_POOL, f"{PARENT_ID}: {decision}")
                print(f"✅ Judgement pushed for: {query}")
                break

            except Exception as e:
                print(f"⚠️ Judge parsing error: {e}")
                continue

        time.sleep(10)

    except Exception as e:
        print(f"❌ Judge error: {e}")
        time.sleep(10)
