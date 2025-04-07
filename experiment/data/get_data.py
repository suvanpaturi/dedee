from datasets import load_dataset
from datasets import concatenate_datasets
import pandas as pd

eli5 = "sentence-transformers/eli5"
squad = "rajpurkar/squad"
hotpotqa = "hotpot_qa"

class QA_Pair:
    def __init__(self, question, answer, source):
        self.question = question
        self.answer = answer,
        self.source = source

def eli5_to_csv():
    eli5_dataset = load_dataset(eli5, "pair")['train']
    data = pd.DataFrame(eli5_dataset)
    data['source'] = eli5
    data.to_csv('./experiment/data/extracted/eli5.csv', index=False)
    
def squad_to_csv():
    data = []
    squad_dataset = load_dataset(squad)
    squad_dataset = concatenate_datasets([squad_dataset['train'], squad_dataset['validation']])
    for s in squad_dataset:
        data.append({'question': s['question'], 'answer': s['answers']['text'][0]})
    data = pd.DataFrame(data)
    data['source'] = squad
    data.to_csv('./experiment/data/extracted/squad.csv', index=False)
    
def hotpotqa_to_csv():
    data = []
    hotpot_dataset = load_dataset(hotpotqa, "distractor")
    hotpot_dataset = concatenate_datasets([hotpot_dataset['train'], hotpot_dataset['validation']])
    for h in hotpot_dataset:
        data.append({'question': h['question'], 'answer': h['answer']})
    data = pd.DataFrame(data)
    data['source'] = hotpotqa
    data.to_csv('./experiment/data/extracted/hotpotqa.csv', index=False)

#------RUN-------
#eli5_to_csv()
#squad_to_csv()
#hotpotqa_to_csv()