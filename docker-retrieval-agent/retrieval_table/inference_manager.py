from tree_retrieval import TreeRetriever
from global_llm import GlobalLLM
from debate import *
from latency_tracker import global_times

class InferenceManager:
    def __init__(self):
        self.retriever = TreeRetriever()
        self.llm = GlobalLLM()

    async def run(self, input):
        
        query = input.query
        model = input.model
        judge_model = input.judge_model
        
        global_times["tree_retrieval"]["start_time"] = time.perf_counter()
        retrieved_knowledge = self.retriever.comprehensive_search(query)
        global_times["tree_retrieval"]["end_time"] = time.perf_counter()
        print("retrieved_knowledge", retrieved_knowledge)
        if not retrieved_knowledge:
            global_times["global_llm"]["start_time"] = time.perf_counter()
            res = await self.llm.invoke(query, None)
            global_times["global_llm"]["end_time"] = time.perf_counter()
            return (res, 'global-llm')
        if len(retrieved_knowledge) == 1:
            return (retrieved_knowledge[0]["answer_text"], 'exact-match')
        global_times["debate"]["start_time"] = time.perf_counter()
        debate_request = DebateRequest(
            query=query,
            model=model,
            judge_model=judge_model,
            total_rounds=3,
            retrieved_knowledge=retrieved_knowledge
        )
        print("debate_request", debate_request)
        final_verdict = await start_debate(request=debate_request)
        print("final_verdict", final_verdict)
        global_times["debate"]["end_time"] = time.perf_counter()
        if final_verdict:
            return (final_verdict, 'debate')
        else:
            global_times["global_llm"]["start_time"] = time.perf_counter()
            res = await self.llm.invoke(query, None)
            global_times["global_llm"]["end_time"] = time.perf_counter()
            return (res, 'global-llm')