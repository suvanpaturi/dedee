from openai import OpenAI
from dotenv import load_dotenv
client = OpenAI()

load_dotenv()

class GlobalLLM:
    def __init__(self):
        self.model = "gpt-4o"
        self.client = OpenAI()
        
    def invoke(self, query, knowledge):
        messages = self.build_messages(query, knowledge)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=20,
            messages=messages
        )
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None
        
    def build_messages(self, query, knowledge):
        messages = [
            {"role": "system", "content": "You are a helpful QA assistant. \
            Answer the following question. You may be provided some additional context in the form of similar \
            question and answer pairs. Only consider these in formulating your response if helpful. Lastly,  \
            provide a concise, yet effective answer. Do not make anything up."}
        ]
        messages.append({"role": "user", "content": "Here is the question: " + query})
        if knowledge:
            knowledge_block = "\n\n".join([
                f"Q: {ex['query']}\nA: {ex['response']}" for ex in knowledge
            ])
            messages.append({"role": "user", "content": "Here are similar question and answer pairs: " + knowledge_block})
        return messages