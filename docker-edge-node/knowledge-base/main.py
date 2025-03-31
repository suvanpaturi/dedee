from fastapi import FastAPI, Body
from pydantic import BaseModel
from knowledge_base import Knowledge, KnowledgeBase
from typing_extensions import Annotated
from sync.sync_manager import SyncManager
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, BackgroundTasks, Body

app = FastAPI()

collection_name = "knowledge-base"
kb = KnowledgeBase(collection_name=collection_name)
sm = SyncManager()

class Query(BaseModel):
    query: str

class KnowledgeBatch(BaseModel):
    items: list[Knowledge]

@app.post("/update/")
async def update_knowledge(
    data: Annotated[KnowledgeBatch, 
                    Body(description="List of knowledge items to update"),
                    ],
    background_tasks: BackgroundTasks):
    try:
        added_data = await run_in_threadpool(kb.update, data.items)
        background_tasks.add_task(sm.send_knowledge, added_data)
        return {"message": "Data successfully updated and sent to global graph"}
    except Exception as e:
        return {'Error has occured': str(e)}

@app.post("/get/", deprecated=True)
async def get(input: Query):
    try:
        data = kb.get(query=input.query)
        return {"data": data}
    except Exception as e:
        return {'Error has occured': str(e)}

@app.get("/peek/")
async def get(n: int):
    try:
        data = kb.peek(n)
        return {"data": data}
    except Exception as e:
        return {'Error has occured': str(e)}
    
@app.post("/clear_kb/")
async def handle_clear_kb():
    try:
        kb.clear_kb()
        return {"message": "Knowledge Base cleared"}
    except Exception as e:
        return {'Error has occured': str(e)}
