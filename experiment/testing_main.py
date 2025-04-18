import asyncio
import time
import json
import httpx
import os
from datetime import datetime
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
)
 
BASE_URL = "http://retrieval-agent-traffic.trafficmanager.net:5001"
#BASE_URL = "http://localhost:5001"
 
async def query_retrieval_agent(client: httpx.AsyncClient, query: str, model: str, progress: Progress):
    """Send a query to the retrieval agent and measure the latency."""
    task = progress.add_task(f"Querying: {query}", start=False)
    progress.start_task(task)
    start_time = time.perf_counter()
    result = await client.post(f"{BASE_URL}/query/", json={"query": query, "model": model, "judge_model": "mistral:7b"})
    result = result.json()
    end_time = time.perf_counter()
    progress.stop_task(task)
    overall_latency = end_time - start_time
    return {
        "query": query,
        "response": result.get('response', "No Response"),
        "method": result.get('method', "unknown"),
        "latency": result.get('latency', {}),
        "overall_latency": overall_latency
        }
 
async def main():
 
    llm_name = "qwen2.5:3b"
    dataset_name = "finqa"
    test_json_path = '/Users/siveshkannan/Documents/VSCode/dedee_main/dedee/experiment/data/test/finqa/testset.json'
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_dict = {d["query"]: d for d in data}
    
    timeout = httpx.Timeout(300.0)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f'./experiment/response/{llm_name}/{dataset_name}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    query_results = []
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            for item in data:
                result = await query_retrieval_agent(client, item["query"], llm_name, progress)
                
                query = result["query"]
                query_result = {
                    "query": query,
                    "predicted_response": result["response"],
                    "method": result["method"],
                    "actual_response": data_dict[query]["response"],
                    "source": data_dict[query]["source"],
                    "latency": result["latency"],
                    "overall_latency": result["overall_latency"]
                }
                query_results.append(query_result)
    
    consolidated_results = {
        "metadata": {
            "llm_name": llm_name,
            "dataset_name": dataset_name,
            "total_queries": len(query_results),
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": timestamp
        },
        "results": query_results
    }
    
    with open(f'{output_dir}/all_results.json', 'w', encoding='utf-8') as f:
        json.dump(consolidated_results, f, ensure_ascii=False, indent=4)
    
    print(f"Testing complete. Processed {len(query_results)} queries.")
    print(f"Results saved to {output_dir}")
    print(f"Timestamp: {timestamp}")
 
if __name__ == "__main__":
    asyncio.run(main())