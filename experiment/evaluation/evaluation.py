
import json
import asyncio
from pathlib import Path
import numpy as np
import importlib.util
import string
import re
import time
from collections import Counter
from evaluate import load

bert = load("bertscore")
rouge = load("rouge")
bleurt = load("bleurt", checkpoint="bleurt-large-512")

spec = importlib.util.spec_from_file_location('helper_llm', './experiment/helper_llm.py')
helper_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_llm)
    
prompt = """
    You are an evaluator whos job is to see if a given answer is a suitable response to a given question.
    You will evaluate based on:
    - relevancy
    - correctness
    - similarity to actual answer
    Give a rating from 1 to 5.
    5 = answer is relevant, answers the given query, and adds to actual answer. It is a good quality answer.
    3 = answer is relevant and somewhat answers query, but is not a good quality answer.
    1 = answer is not relevant and does not answer the given query.
    - 2 and 4 are in between.
    
    Provide ONLY the rating.
    
    Given question: {question}
    Given answer: {answer}
    
    Actual answer: {actual_answer}
"""

llm = helper_llm.HelperLLM()
llm.set_prompt(prompt=prompt, type="evaluation")

def get_bert_scores(predictions, actual):
    bert_scores = bert.compute(predictions=predictions, references=actual, lang="en")
    return bert_scores

def get_rouge_scores(predictions, actual):
    scores = []
    for pred, ref in zip(predictions, actual):
        result = rouge.compute(predictions=[pred], references=[ref])
        scores.append(result["rougeL"])
    return scores
    
def get_bleurt_scores(predictions, actual):
    bleurt_scores = bleurt.compute(predictions=predictions, references=actual)
    return bleurt_scores["scores"]

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
    file_path = file_path.resolve()
    anchor = Path("experiment/response").resolve()

    relative = file_path.relative_to(anchor)
    model, dataset, _, filename = relative.parts
    return Path("experiment/evaluation") / model / dataset / filename

def evaluate(file_path: Path):
    eval_results = []
    
    print(f"Evaluating: {file_path}")
    time.sleep(5)

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
            "source": item["source"],
            "latency": calculate_latencies(item["latency"]),
            "overall_latency": item["overall_latency"],
        })
    
    
    #NEED TO FIX
    rating_results = llm.process_batch(data=eval_results)
    if len(rating_results) == len(eval_results):
        for i, result in enumerate(rating_results):
            eval_results[i]["score"] = result["score"]

    predictions = [item["predicted_response"] for item in eval_results]
    references = [item["actual_response"] for item in eval_results]
    
    bert_scores = get_bert_scores(predictions, references)["f1"]
    rouge_scores = get_rouge_scores(predictions, references)
    bleurt_scores = get_bleurt_scores(predictions, references)

    # Token-overlap F1
    f1_scores = [get_f1_score(pred, ref)[0] for pred, ref in zip(predictions, references)]
    
    for i, (bert, f1, rogue, bleurt) in enumerate(zip(bert_scores, f1_scores, rouge_scores, bleurt_scores)):
        eval_results[i]["bert_score"] = bert
        eval_results[i]["f1_score"] = f1
        eval_results[i]["rouge_score"] = rogue
        eval_results[i]["bleurt_score"] = bleurt
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": eval_results}, f, indent=2)

path = Path("/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/qwen2.5:3b/squad/20250419_155529/all_results.json")

if __name__ == "__main__":
    evaluate(path)