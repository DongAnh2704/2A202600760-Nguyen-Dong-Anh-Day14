import asyncio
import time
from typing import List, Dict, Any
from engine.retrieval_eval import RetrievalEvaluator

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator # RetrievalEvaluator
        self.judge = judge         # LLMJudge

    def _calculate_cost(self, model_usages: List[Dict]) -> float:
        cost = 0.0
        for usage in model_usages:
            model = usage.get("model", "")
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            
            # Pricing rules
            if "gemini-3.5-flash" in model or "flash" in model:
                cost += p_tokens * 0.000000075 + c_tokens * 0.00000030
            elif "gemini-2.5-flash" in model:
                cost += p_tokens * 0.000000075 + c_tokens * 0.00000030
            elif "gemini-2.5-pro" in model or "pro" in model:
                cost += p_tokens * 0.00000125 + c_tokens * 0.00000500
            else:
                cost += p_tokens * 0.000000075 + c_tokens * 0.00000030
        return cost

    async def run_single_test(self, test_case: Dict, semaphore: asyncio.Semaphore) -> Dict:
        async with semaphore:
            start_time = time.perf_counter()
            
            # 1. Query the Agent
            response = await self.agent.query(test_case["question"])
            latency = time.perf_counter() - start_time
            
            # Extract retrieved IDs
            retrieved_ids = response.get("metadata", {}).get("retrieved_ids", [])
            expected_ids = test_case.get("expected_retrieval_ids", [])
            
            # 2. Local Retrieval Metrics
            hit_rate = self.evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.evaluator.calculate_mrr(expected_ids, retrieved_ids)
            
            # 3. Call Multi-Judge
            judge_result = await self.judge.evaluate_multi_judge(
                test_case["question"], 
                response["answer"], 
                test_case["expected_answer"]
            )
            
            # 4. Token & Cost usage tracking
            agent_prompt_tokens = response.get("metadata", {}).get("prompt_tokens", 0)
            agent_completion_tokens = response.get("metadata", {}).get("completion_tokens", 0)
            
            model_usages = [
                {
                    "model": response.get("metadata", {}).get("model", "gemini-3.5-flash"),
                    "prompt_tokens": agent_prompt_tokens,
                    "completion_tokens": agent_completion_tokens
                }
            ]
            
            # Append judge usages
            judge_usages = judge_result.get("model_usages", [])
            model_usages.extend(judge_usages)
            
            total_prompt_tokens = agent_prompt_tokens + judge_result.get("prompt_tokens", 0)
            total_completion_tokens = agent_completion_tokens + judge_result.get("completion_tokens", 0)
            
            cost = self._calculate_cost(model_usages)
            
            # Only sleep to rate limit if we actually made successful API calls
            agent_success = agent_prompt_tokens > 0
            judge_success = judge_result.get("prompt_tokens", 0) > 0
            if agent_success or judge_success:
                await asyncio.sleep(8)
            
            return {
                "test_case": test_case["question"],
                "agent_response": response["answer"],
                "expected_answer": test_case["expected_answer"],
                "latency": latency,
                "cost": cost,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "tokens_used": total_prompt_tokens + total_completion_tokens,
                "ragas": {
                    "retrieval": {
                        "hit_rate": hit_rate,
                        "mrr": mrr
                    }
                },
                "judge": {
                    "final_score": judge_result["final_score"],
                    "agreement_rate": judge_result["agreement_rate"],
                    "reasoning": judge_result["reasoning"],
                    "individual_scores": judge_result["individual_scores"]
                },
                "status": "fail" if judge_result["final_score"] < 3 else "pass",
                "metadata": test_case.get("metadata", {})
            }

    async def run_all(self, dataset: List[Dict], batch_size: int = 1) -> List[Dict]:
        """
        Runs the benchmark with parallel tasks limited by batch_size semaphore.
        Uses batch_size = 1 or 2 to stay strictly under Gemini free tier limit.
        """
        # Concurrency semaphore
        sem = asyncio.Semaphore(batch_size)
        
        tasks = [self.run_single_test(case, sem) for case in dataset]
        results = await asyncio.gather(*tasks)
        return results
