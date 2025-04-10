from tree_retrieval import TreeRetriever
from global_llm import GlobalLLM
from debate import *
from latency_tracker import global_times

class InferenceManager:
    def __init__(self):
        self.retriever = TreeRetriever()
        self.llm = GlobalLLM()

    async def run(self, query):
        global_times["tree_retrieval"]["start_time"] = time.perf_counter()
        retrieved_knowledge = self.retriever.comprehensive_search(query)
        print("retrieved_knowledge", retrieved_knowledge)
        if len(retrieved_knowledge) == 1:
            return retrieved_knowledge[0]["answer_text"]
        global_times["tree_retrieval"]["end_time"] = time.perf_counter()
        if not retrieved_knowledge:
            global_times["global_llm"]["start_time"] = time.perf_counter()
            res = await self.llm.invoke(query, None)
            global_times["global_llm"]["end_time"] = time.perf_counter()
            return res
        global_times["debate"]["start_time"] = time.perf_counter()
        print("retrieved_knowledge", retrieved_knowledge)
        debate_request = DebateRequest(
            query=query,
            total_rounds=3,
            retrieved_knowledge=retrieved_knowledge
        )
        print("debate_request", debate_request)
        final_verdict = await start_debate(request=debate_request)
        print("final_verdict", final_verdict)
        global_times["debate"]["end_time"] = time.perf_counter()
        if final_verdict:
            return final_verdict
        else:
            global_times["global_llm"]["start_time"] = time.perf_counter()
            res = await self.llm.invoke(query, None)
            global_times["global_llm"]["end_time"] = time.perf_counter()
            return res