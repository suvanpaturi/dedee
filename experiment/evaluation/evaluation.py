
import json
import os
import numpy as np
import importlib.util
import string
import re
from collections import Counter
from evaluate import load

bert = load("bertscore")
rouge = load("rouge")
meteor = load("meteor")

spec = importlib.util.spec_from_file_location('helper_llm', './experiment/helper_llm.py')
helper_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_llm)

with open('./experiment/response/query_results.json', 'w', encoding='utf-8') as f:
        query_results = json.load(f, ensure_ascii=False, indent=4)
    
prompt = """
    You are an evaluator. Given a question and an answer, rate on a scale of 0 to 5 if the 
    question is relevant or answers the given query. 5 means the answer answers the question, while 0
    means it does not. Provide only the rating and do not offer any additional text.x   
    
    Given question: {question}
    Given answer: {answer}
"""

llm = helper_llm.HelperLLM()
llm.set_prompt(prompt=prompt, type="evaluation")
results = llm.process_batch(data=query_results)
print(results)

def get_calculated_scores(data):
    '''
    bert_score = bert.compute(predictions=[predictions], references=references, lang="en")
    rouge_score = rouge.compute(predictions=[predictions], references=references)
    meteor_score = meteor.compute(predictions=[predictions], references=references)
    '''
    return data

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

'''
with open('./experiment/data/test/testset.json', 'w', encoding='utf-8') as f:
    json.dump(final_testdata.tolist(), f, ensure_ascii=False, indent=4)
'''
