import importlib.util
import json

spec = importlib.util.spec_from_file_location('helper_llm', './experiment/helper_llm.py')
helper_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_llm)
    
prompt = prompt = """
    You are an evaluator. Rate how well a given answer responds to a question, using these criteria:
    - Relevance to the question
    - Correctness
    - Similarity to the actual answer

    Rate from 1 to 5:
    5 = Fully answers the question, matches or improves on the actual answer (extra correct info is okay).
    3 = Partially answers the question, but lacks clarity or quality.
    1 = Irrelevant or incorrect.

    Use 2 or 4 if the answer is in between.

    Do NOT penalize for extra info if:
    - The key points match the actual answer
    - The extra info is reasonable and not contradictory

    Example:
    Q: What are the two parts of a cell?
    A: The cell has many parts including the nucleus and mitochondria. But the two main parts are the nucleus and cytoplasm.
    Actual: nucleus and cytoplasm
    → Score: 5

    ONLY return the number.

    Question: {question}
    Answer: {answer}
    Actual answer: {actual_answer}
"""


llm = helper_llm.HelperLLM()
llm.set_prompt(prompt=prompt, type="evaluation")
    
def evaluate_llm(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    eval_results = data.get("results", [])
    rating_results = llm.process_batch(data=eval_results)
    if len(rating_results) != len(eval_results):
        raise ValueError("Mismatch between rating and eval results.")
    if len(rating_results) == len(eval_results):
        for i, result in enumerate(rating_results):
            eval_results[i]["score"] = result["score"]
    data["results"] = eval_results
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    

#----------FINAQA-------------
#gemma2b 
path_finqa_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/gemma:2b/finqa/all_results.json"
#phi3:3b
path_finqa_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/phi3:3.8b/finqa/all_results.json"
#qwen2.5:3b
path_finqa_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/qwen2.5:3b/finqa/all_results.json"


#----------HOTPOTQA-------------
#gemma2b 
path_hotpota_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/gemma:2b/hotpotqa/all_results.json"
#phi3:3b
path_hotpota_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/phi3:3.8b/hotpotqa/all_results.json"
#qwen2.5:3b
path_hotpota_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/qwen2.5:3b/hotpotqa/all_results.json"


#----------SQUAD-------------
#gemma2b 
path_squad_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/gemma:2b/squad/all_results.json"
#phi3:3b
path_squad_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/phi3:3.8b/squad/all_results.json"
#qwen2.5:3b
path_squad_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/qwen2.5:3b/squad/all_results.json"


#----------ScienceQA-------------

#gemma2b 
path_scienceqa_gemma = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/gemma:2b/science_qa/all_results.json"
#phi3:3b
path_scienceqa_phi3 = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/phi3:3.8b/science_qa/all_results.json"
#qwen2.5:3b
path_scienceqa_qwen = "/Users/suvanpaturi/Library/CloudStorage/GoogleDrive-suvan.paturi@gmail.com/My Drive/Research/dedee/experiment/evaluation/qwen2.5:3b/science_qa/all_results.json"

if __name__ == "__main__":
    #evaluate_llm(path_finqa_gemma)
    evaluate_llm(path_finqa_phi3)
    #evaluate_llm(path_finqa_qwen)
    
    #evaluate_llm(path_hotpota_gemma)
    #evaluate_llm(path_hotpota_phi3)
    #evaluate_llm(path_hotpota_qwen)
    
    #evaluate_llm(path_squad_gemma)
    #evaluate_llm(path_squad_phi3)
    #evaluate_llm(path_squad_qwen)
    
    #evaluate_llm(path_scienceqa_gemma)
    #evaluate_llm(path_scienceqa_phi3)
    #evaluate_llm(path_scienceqa_qwen)
    