from datasets import load_dataset
import pandas as pd

# Load the first 10,000 rows
dataset = load_dataset("KonstantyM/science_qa", split="train[:10000]")

rows = []
for item in dataset:
    query = item.get("question", "").strip()
    response = item.get("answer", "").strip()
    source = "science_qa"
    rows.append({"query": query, "response": response, "source": source})

df = pd.DataFrame(rows)
df.to_csv("science_qa_first_10000.csv", index=False)
