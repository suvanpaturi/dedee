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

@app.get("/query/")
async def handle_query(input: Query):
    try:
        response = None
        results = table.query(query=input.query)
        if results:
            response = results["response"]
        return {"query": input.query, "response": response}
    except Exception as e:
        return {'Error has occured': str(e)}

@app.post("/put/")
async def handle_put(input: QueryResponse):
    try:
        table.put(query=input.query, response=input.response)
        return {"query": input.query, "response": input.response}
    except Exception as e:
        return {'Error has occured': str(e)}
    
@app.post("/clear_table/")
async def handle_clear_table():
        try:
            table.clear_table()
            return {"message": "Table cleared"}
        except Exception as e:
            return {'Error has occured': str(e)}
