from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import get_openai_callback
import os
import time
from openai._exceptions import RateLimitError

api_key = os.environ.get("OPENAI_API_KEY")

class HelperLLM:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=50,
            api_key=api_key
        )
        self.chain = None
        self.prompt_type=None
        
    def set_prompt(self, prompt, type):
        template = ChatPromptTemplate.from_template(prompt)
        self.chain = template | self.model | StrOutputParser()
        self.prompt_type = type
        
    def process(self, item):
        try:
            if self.chain and self.prompt_type:
                query = item['query']
                response = item['predicted_response']
                actual_response = item['actual_response']
                if self.prompt_type == 'evaluation':
                    with get_openai_callback() as cb:
                        result = self.chain.invoke({"question": query, "answer": response, "actual_answer": actual_response})
                        return {
                            "query": query,
                            "predicted_response": response,
                            "actual_response": actual_response,
                            "score": result,
                        }
                if self.prompt_type == "generation":
                    query = item['query']
                    response = item['response']
                    source = item['source']
                    with get_openai_callback() as cb:
                        result = self.chain.invoke({"question": query})
                        return {
                            "query": query,
                            "response": response,
                            "source": source,
                            "modified_query": result
                        }
                return {}
        except Exception as e:
            return {}
        
    def process_batch(self, data, batch_size=60, max_attempts=5):
        all_results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            if self.prompt_type == 'evaluation':
                inputs = [{"question": item["query"], "answer": item["predicted_response"], "actual_answer": item["actual_response"]} for item in batch]
            if self.prompt_type == "generation":
                inputs = [{"question": item["query"]} for item in batch]
            
            with get_openai_callback() as cb:
                try:
                    for _ in range(max_attempts):
                        try:
                            batch_results = self.chain.batch(inputs)
                            for j, result in enumerate(batch_results):
                                if self.prompt_type == 'evaluation':
                                    all_results.append({
                                        "query": batch[j]["query"],
                                        "predicted_response": batch[j]["predicted_response"],
                                        "actual_response": batch[j]["actual_response"],
                                        "score": result
                                    })
                                if self.prompt_type == "generation":
                                    all_results.append({
                                        "query": batch[j]["query"],
                                        "response": batch[j]["response"],
                                        "source": batch[j]["source"],
                                        "modified_query": result
                                    })
                            break
                        except RateLimitError:
                            print("Rate limit error, retrying...")
                            time.sleep()
                            continue
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
            print(f"Processed batch {i//batch_size + 1}/{(len(data)+batch_size-1)//batch_size}")
         
        return all_results
            
        