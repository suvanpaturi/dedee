from fastapi import FastAPI
from pydantic import BaseModel
from table import RetrievalTable
from inference_manager import InferenceManager
from latency_tracker import global_times
import time

app = FastAPI()

collection_name = "retrieval-table"
table = RetrievalTable(collection_name=collection_name)
inference_manager = InferenceManager()

class Query(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str

@app.post("/query/")
async def handle_query(input: Query):
    try:
        global_times["retrieval_table"]["start_time"] = time.perf_counter()
        results = table.query(query=input.query)
        global_times["retrieval_table"]["end_time"] = time.perf_counter()
        if results:
            response = results["response"]
            return {"query": input.query, "response": response, "latency": global_times}
        else:
            inference_response = inference_manager.run(input.query)
            if inference_response:
                table.put(query=input.query, response=inference_response)
            return {"query": input.query, "response": inference_response, "latency": global_times}
    except Exception as e:
        return {'Error has occurred': str(e)}

@app.post("/put/")
async def handle_put(input: QueryResponse):
    try:
        table.put(query=input.query, response=input.response)
        return {"query": input.query, "response": input.response}
    except Exception as e:
        return {'Error has occurred': str(e)}

@app.post("/clear_table/")
async def handle_clear_table():
    try:
        table.clear_table()
        return {"message": "Table cleared"}
    except Exception as e:
        return {'Error has occurred': str(e)}

@app.get("/ping/")
async def ping():
    return {"status": "ok"}