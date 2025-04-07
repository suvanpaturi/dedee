import asyncio
import time
import json
import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn

BASE_URL = "http://retrieval-agent-traffic.trafficmanager.net:5001"

async def query_retrieval_agent(client: httpx.AsyncClient, query: str, progress: Progress):
    """Send a query to the retrieval agent and measure the latency."""
    task = progress.add_task(f"Querying: {query}", start=False)
    progress.start_task(task)

    start_time = time.perf_counter()
    result = await client.post(f"{BASE_URL}/query/", json={"query": query})
    result = result.json()
    end_time = time.perf_counter()

    progress.stop_task(task)
    latency = end_time - start_time
    return {
        "query": query, 
        "response": result.get('response', "No Response"),
        "latency": latency}

async def main():
    
    ''' UNCOMMENT FOR FULL TESTING
    with open('./experiment/test/testset.json', 'r', encoding='utf-8') as f:
        data = json.load(f, ensure_ascii=False, indent=4)
    '''
    
    data = [
        {
            "query": "In which city was the composer of \"Prima la musica e poi le parole\" born?",
            "response": "Legnago",
            "source": "hotpot_qa"
        },
        {
            "query": "I Am Your Gummy Bear is the debut album by what German multilingual character?",
            "response": "Gummibär",
            "source": "hotpot_qa"
        }
    ]
    
    data_dict = {d["query"]: d for d in data} #store test data in dict
    
    async with httpx.AsyncClient() as client:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            tasks = [query_retrieval_agent(client, item["query"], progress) for item in data]
            results = await asyncio.gather(*tasks)
    
    query_results = []
    for r in results:
        query = r["query"]
        query_results.append({
            "query": query,
            "predicted_response": r["response"],
            "actual_response": data_dict[query]["response"],
            "source": data_dict[query]["source"],
            "latency": r["latency"]
        })
      
    with open('./experiment/response/query_results.json', 'w', encoding='utf-8') as f:
        json.dump(query_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(main())