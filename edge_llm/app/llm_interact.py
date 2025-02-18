from fastapi import FastAPI
from langchain.prompts import PromptTemplate 
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from llm import llm_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

template = """You are a very smart and educated assistant to guide the user to understand the concepts. Please Explain the answer
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Only return the helpful answer below and nothing else. Give an answer in 1000 characteres at maximum please
Helpful answer:
"""
prompt = PromptTemplate.from_template(template) 

async def proces_query(payload_json):
    try:
        question = payload_json.get("question")
        complete_prompt = prompt.format(question=question)
        response = llm_pipeline(complete_prompt)
        return response[:1000]
    except Exception as e:
        x = 1

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5005)
