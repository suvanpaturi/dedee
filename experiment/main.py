from models.query import UserQuery
from rich.progress import Progress, SpinnerColumn, TextColumn
import httpx
import asyncio

BASE_URL = "http://retrieval-agent-traffic.trafficmanager.net:5001"

async def query_retrieval_agent(query: str):
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Querying retrieval agent...", start=False)
        progress.start_task(task)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/query/", params={"query": query})
            progress.stop()
            return response.json()

async def main():
    result = await query_retrieval_agent("What is the capital of France?")
    print(result)

asyncio.run(main())

