
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import string
import re
import time
from collections import Counter
from evaluate import load

bert = load("bertscore")
rouge = load("rouge")
bleurt = load("bleurt", "bleurt-large-512")

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

def evaluate(file_path: str):
    eval_results = []
    file_path = Path(file_path)
    print(f"Evaluating: {file_path}")
    
    steps = [
        "Reading file",
        "Calculating latencies",
        "Calculating BERT",
        "Calculating Rouge",
        "Calculating BLEURT",
        "Calculating F1,"
        "Writing scores to file"
    ]
    
    pbar = tqdm(total=len(steps), desc="Evaluating", bar_format="{l_bar}{bar} [Step {n_fmt}/{total_fmt}] {desc}")
    pbar.set_description("Reading file")

    time.sleep(5)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])

    output_path = convert_to_eval_path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pbar.update(1)

    
    pbar.set_description("Calculating latencies")

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
    pbar.update(1)
    
    predictions = [item["predicted_response"] for item in eval_results]
    references = [item["actual_response"] for item in eval_results]
    
    pbar.set_description("Calculating BERT")
    bert_scores = get_bert_scores(predictions, references)["f1"]
    pbar.update(1)
    
    pbar.set_description("Calculating Rouge")
    rouge_scores = get_rouge_scores(predictions, references)
    pbar.update(1)
    
    pbar.set_description("Calculating BLEURT")
    bleurt_scores = get_bleurt_scores(predictions, references)
    pbar.update(1)
    
    # Token-overlap F1
    pbar.set_description("Calculating F1")
    f1_scores = [get_f1_score(pred, ref)[0] for pred, ref in zip(predictions, references)]
    pbar.update(1)
    
    pbar.update(1)
    print("BERT Scores: ", len(bert_scores))
    print("F1 Scores: ", len(f1_scores))
    print("Rouge Scores: ", len(rouge_scores))
    print("BLEURT Scores: ", len(bleurt_scores))
    
    pbar.set_description("Writing scores to file")

    for i in range(len(eval_results)):
        eval_results[i]["bert_score"] = bert_scores[i]
        eval_results[i]["f1_score"] = f1_scores[i]
        eval_results[i]["rouge_score"] = rouge_scores[i]
        eval_results[i]["bleurt_score"] = bleurt_scores[i]
               
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": eval_results}, f, indent=2)
        
    pbar.update(1)
    pbar.close()
#----------FINAQA-------------
#gemma2b 
path_finqa_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/gemma:2b/finqa/20250418_102834/all_results.json"
#phi3:3b
path_finqa_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/phi3:3.8b/finqa/20250424_234434/all_results.json"
#"/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/phi3:3.8b/finqa/20250418_123737/all_results.json"

#qwen2.5:3b
path_finqa_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/qwen2.5:3b/finqa/20250418_144247/all_results.json"


#----------HOTPOTQA-------------
#gemma2b 
path_hotpota_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/gemma:2b/hotpotqa/20250418_215412/all_results.json"
#phi3:3b
path_hotpota_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/phi3:3.8b/hotpotqa/20250418_191823/all_results.json"
#qwen2.5:3b
path_hotpota_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/qwen2.5:3b/hotpotqa/20250418_173120/all_results.json"


#----------SQUAD-------------
#gemma2b 
path_squad_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/gemma:2b/squad/20250419_211347/all_results.json"
#phi3:3b
path_squad_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/phi3:3.8b/squad/20250419_220937/all_results.json"
#qwen2.5:3b
path_squad_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/qwen2.5:3b/squad/20250419_235112/all_results.json"


#----------ScienceQA-------------

#gemma2b 
path_scienceqa_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/gemma:2b/science_qa/20250420_001136/all_results.json"
#phi3:3b
path_scienceqa_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/phi3:3.8b/science_qa/20250421_100032/all_results.json"
#qwen2.5:3b
path_scienceqa_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/response/qwen2.5:3b/science_qa/20250420_172413/all_results.json"

if __name__ == "__main__":
    #evaluate(path_finqa_gemma)
    evaluate(path_finqa_phi3)
    #evaluate(path_finqa_qwen)
    
    #evaluate(path_hotpota_gemma)
    #evaluate(path_hotpota_phi3)
    #evaluate(path_hotpota_qwen)
    
    #evaluate(path_squad_gemma)
    #evaluate(path_squad_phi3)
    #evaluate(path_squad_qwen)
    
    #evaluate(path_scienceqa_gemma)
    #evaluate(path_scienceqa_phi3)
    #evaluate(path_scienceqa_qwen)