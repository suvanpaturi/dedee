from fastapi import FastAPI
from pydantic import BaseModel
from table import RetrievalTable

app = FastAPI()

collection_name = "retrieval-table"
table = RetrievalTable(collection_name=collection_name)

class Query(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str

@app.post("/query/")
async def handle_query(input: Query):
    response = None
    results = table.query(query=input.query)
    if results:
        response = results["response"]
    return {"query": input.query, "response": response}

@app.post("/put/")
async def handle_put(input: QueryResponse):
    table.put(query=input.query, response=input.response)
    return {"query": input.query, "response": input.response}

@app.get("/clear_table/")
async def handle_clear_table():
    table.clear()
    return {"message": "Table cleared"}
