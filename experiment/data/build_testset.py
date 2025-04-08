import json
import os
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location('helper_llm', './experiment/helper_llm.py')
helper_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_llm)

seed = 17

edge_data_path = './experiment/data/edge'
edge_data = []

np.random.seed(seed)

for file in os.listdir(edge_data_path):
    if file.endswith('.json') and file.startswith('edge-device'):
        file_path = os.path.join(edge_data_path, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                edge_data.append(item)
            

edge_data = np.array(edge_data)
test_data = np.random.choice(edge_data, size=500, replace=False)
np.random.shuffle(test_data)

split_index = int(len(test_data) * 0.7)
exact_test_data = test_data[:split_index]
altered_test_data = test_data[split_index:]

with open('./experiment/data/test/exact_testset.json', 'w', encoding='utf-8') as f:
    json.dump(exact_test_data.tolist(), f, ensure_ascii=False, indent=4)
    
prompt = """
    Reword the following question in a different way, keeping the same meaning and expected answer. 
    Be concise and avoid adding extra information. Return only the reworded question. Do not lose helpful context
    in rewording.
   
    Ex. What is the capital of France? -> Which city is France's capital?
    
    Given question: {question}
"""

llm = helper_llm.HelperLLM()
llm.set_prompt(prompt=prompt, type="generation")
results = llm.process_batch(data=altered_test_data)

with open('./experiment/data/test/altered_testset.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

set_results = [{
            "query": item["modified_query"],
            "response": item["response"],
            "source": item["source"]
        }
        for item in results]

final_testdata = np.concatenate([exact_test_data, set_results])

with open('./experiment/data/test/testset.json', 'w', encoding='utf-8') as f:
    json.dump(final_testdata.tolist(), f, ensure_ascii=False, indent=4)





    