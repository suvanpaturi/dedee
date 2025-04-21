from pathlib import Path
import json
import pandas as pd

root_folder = Path("./experiment/evaluation")

all_cases = []
debate_cases = []

def get_metrics(dataset, model, results):
    average_overall_latency = sum([result['overall_latency'] for result in results]) / len(results)
    average_debate_latency = sum([result['latency']['debate'] for result in results]) / len(results)
    average_tree_latency = sum([result['latency']["tree_retrieval"] for result in results]) / len(results)
    average_f1 = sum([result[ "f1_score"] for result in results]) / len(results)
    average_bert_score = sum([result["bert_score"] for result in results]) / len(results)
    average_rogue_score = sum([result["rouge_score"] for result in results]) / len(results)
    average_bleurt_score = sum([result["bleurt_score"] for result in results]) / len(results)
    average_llm_critic_score = sum([result["score"] for result in results]) / len(results)
    return {
        "dataset": dataset,
        "model": model,
        "average_overall_latency": average_overall_latency,
        "average_debate_latency": average_debate_latency,
        "average_tree_latency": average_tree_latency,
        "average_f1": average_f1,
        "average_bert_score": average_bert_score,
        "average_rogue_score": average_rogue_score,
        "average_bleurt_score": average_bleurt_score,
        "average_llm_critic_score": average_llm_critic_score
    }

def analyze(root_folder):
    all_cases = []
    debate_cases = []
    for subdir in root_folder.iterdir():
        if subdir.is_dir():
            model_name = subdir.name
            for dataset_dir in subdir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    print(f"Processing dataset: {dataset_name} for model: {model_name}")                
                    for file in dataset_dir.glob("*.json"):
                        if file.is_file():
                            print(f"Processing file: {file}")
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                all_results = data.get("results", [])
                                debate_results = [x for x in all_results if x['method'] == 'debate']
                                all_metrics = get_metrics(dataset_name, model_name, all_results)
                                debate_metrics = get_metrics(dataset_name, model_name, debate_results)
                                
                                all_cases.append(all_metrics)
                                debate_cases.append(debate_metrics)
                            
    df = pd.DataFrame(all_cases)
    df.to_csv('./experiment/evaluation/all_cases.csv', index=False)
    df = pd.DataFrame(debate_cases)
    df.to_csv('./experiment/evaluation/debate_cases.csv', index=False)
    
if __name__ == "__main__":
    analyze(root_folder)
    print("Analysis complete. Results saved to CSV files.")