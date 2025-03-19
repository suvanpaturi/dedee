from fastapi import FastAPI, Body
from pydantic import BaseModel
from knowledge_base import Knowledge, KnowledgeBase
from typing_extensions import Annotated
from sync import sync_manager

app = FastAPI()

collection_name = "knowledge-base"
kb = KnowledgeBase(collection_name=collection_name)
sm = sync_manager.GraphSyncManager()

class Query(BaseModel):
    query: str

class KnowledgeBatch(BaseModel):
    items: list[Knowledge]

@app.post("/update/")
async def update_knowledge(
    data: Annotated[KnowledgeBatch, 
                    Body(description="List of knowledge items to update")
                    ]):
    added_data = kb.update(data.items)
    await sm.insert_parallel(added_data)
    return {"message": "Data successfully updated and send to global graph"}

@app.get("/get/")
async def get(input: Query):
    data = kb.get(query=input.query)
    return {"data": data}

@app.get("/clear_kb/")
async def handle_clear_kb():
    kb.clear_kb()
    return {"message": "Knowledge Base cleared"}
