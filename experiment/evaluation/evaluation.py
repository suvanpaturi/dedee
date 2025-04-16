
import json
import asyncio
from pathlib import Path
import numpy as np
import importlib.util
import string
import re
from collections import Counter
from evaluate import load

bert = load("bertscore")

spec = importlib.util.spec_from_file_location('helper_llm', './experiment/helper_llm.py')
helper_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_llm)
    
prompt = """
    You are an evaluator whos job is to see if a given answer is a suitable response to a given question.
    You will evaluate based on:
    - relevancy
    - correctness
    - similarity to actual answer
    Give a rating from 0 to 5.
    5 = answer is relevant and answers the given query. 
    0 = answer is not relevant and does not answer the given query.
    
    Provide only the rating.
    
    Given question: {question}
    Given answer: {answer}
    
    Actual answer: {actual_answer}
"""

llm = helper_llm.HelperLLM()
llm.set_prompt(prompt=prompt, type="evaluation")

def get_bert_scores(predictions, actual):
    bert_scores = bert.compute(predictions=predictions, references=actual, lang="en")
    return bert_scores

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def latest_run(path: Path):
    timestamp_dirs = sorted(
        [p for p in path.iterdir() if p.is_dir()],
        key=lambda x: x.name,
        reverse=True
    )
    return timestamp_dirs[0] if timestamp_dirs else None

def calculate_latencies(latency_data):
    l = {}
    if latency_data:
        for stage in latency_data:
            l[stage] = latency_data[stage]["end_time"] - latency_data[stage]["start_time"]
    return l

def convert_to_eval_path(file_path: Path) -> Path:
    relative = file_path.relative_to("experiment/response")
    model, dataset, _, filename = relative.parts
    return Path("experiment/evaluation") / model / dataset / filename

async def evaluate(file_path: Path):
    eval_results = []
    
    print(f"Evaluating: {file_path}")
    await asyncio.sleep(0.1)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])

    output_path = convert_to_eval_path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for item in results:
        eval_results.append({
            "query": item["query"],
            "predicted_response": item["predicted_response"],
            "method": item["method"],
            "actual_response": item["actual_response"],
            "source": "hotpot_qa",
            "latency": calculate_latencies(item["latency"]),
            "overall_latency": item["overall_latency"],
        })
    
    rating_results = llm.process_batch(data=eval_results)
    for i, result in enumerate(rating_results):
        eval_results[i]["score"] = result["score"]

    await asyncio.sleep(5)

    # Compute BERTScore
    predictions = [item["predicted_response"] for item in eval_results]
    references = [item["actual_response"] for item in eval_results]
    bert_scores = get_bert_scores(predictions, references)  # e.g. returns dict with 'f1'

    # Compute traditional token-overlap F1
    f1_scores = [get_f1_score(pred, ref) for pred, ref in zip(predictions, references)]

    for i, (bert_f1, f1) in enumerate(zip(bert_scores["f1"], f1_scores)):
        eval_results[i]["bert_score_f1"] = bert_f1
        eval_results[i]["f1_score"] = f1
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": eval_results}, f, indent=2)
    
async def run_evaluate(root_dir: str):
    tasks = []
    root = Path(root_dir)
    for model_dir in root.iterdir():
        if model_dir.is_dir():
            for dataset_dir in model_dir.iterdir():
                if dataset_dir.is_dir():
                    latest = latest_run(dataset_dir)
                    if latest:
                        for file in latest.glob("*.json"):
                            tasks.append(asyncio.create_task(evaluate(file)))  
    await asyncio.gather(*tasks)

asyncio.run(run_evaluate("./experiment/response"))