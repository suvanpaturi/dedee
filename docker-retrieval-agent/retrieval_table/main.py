from fastapi import FastAPI
from pydantic import BaseModel
from table import RetrievalTable
from inference_manager import InferenceManager
from latency_tracker import global_times
import time
from debate import ParentResponse, submit_response, get_round_responses

app = FastAPI()

collection_name = "retrieval-table"
table = RetrievalTable(collection_name=collection_name)
inference_manager = InferenceManager()

class Query(BaseModel):
    query: str
    model: str
    judge_model: str

class QueryResponse(BaseModel):
    query: str
    response: str

@app.post("/query/")
async def handle_query(input: Query):
    try:
        global_times["retrieval_table"]["start_time"] = time.perf_counter()
        print(global_times)
        '''
        results = await table.query(query=input.query)
        print("results", results)
        global_times["retrieval_table"]["end_time"] = time.perf_counter()
        if results:
            response = results["response"]
            return {"query": input.query, "response": response, "latency": global_times}
        
        else:
            inference_response = await inference_manager.run(input.query)
            if inference_response:
                await table.put(query=input.query, response=inference_response)
            return {"query": input.query, "response": inference_response, "latency": global_times}
        '''
        global_times["retrieval_table"]["end_time"] = time.perf_counter()
        inference_response, method = await inference_manager.run(input)
        print("inference_response", inference_response)
        return {"query": input.query, "response": inference_response, "method": method, "latency": global_times}
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

@app.post("/submit_response/")
def handle_submit_response(response: ParentResponse):
    return submit_response(response)

@app.get("/round_responses/{round_number}")
def handle_get_round_responses(round_number: int, query_id: str):
    return get_round_responses(round_number, query_id)

@app.get("/ping/")
async def ping():
    return {"status": "ok"}