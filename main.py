import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    # Instantiate real components instead of mock
    agent = MainAgent(version=agent_version)
    evaluator = RetrievalEvaluator()
    judge = LLMJudge()
    
    runner = BenchmarkRunner(agent, evaluator, judge)
    
    # Run all cases (limit batch_size to 1 to stay strictly under rate limits and run successfully)
    results = await runner.run_all(dataset, batch_size=1)

    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    hit_rate = sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total
    mrr = sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total
    agreement_rate = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total
    total_cost = sum(r["cost"] for r in results)
    total_tokens = sum(r["tokens_used"] for r in results)

    summary = {
        "metadata": {
            "version": agent_version, 
            "total": total, 
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "avg_score": avg_score,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "agreement_rate": agreement_rate,
            "avg_latency_sec": avg_latency,
            "total_cost_usd": total_cost,
            "total_tokens_used": total_tokens
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    # 1. Run Agent V1 (Base)
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # 2. Run Agent V2 (Optimized)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    delta_hit_rate = v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"]
    delta_mrr = v2_summary["metrics"]["mrr"] - v1_summary["metrics"]["mrr"]
    
    print(f"V1 Avg Score: {v1_summary['metrics']['avg_score']:.2f} | Hit Rate: {v1_summary['metrics']['hit_rate']:.2f} | MRR: {v1_summary['metrics']['mrr']:.2f}")
    print(f"V2 Avg Score: {v2_summary['metrics']['avg_score']:.2f} | Hit Rate: {v2_summary['metrics']['hit_rate']:.2f} | MRR: {v2_summary['metrics']['mrr']:.2f}")
    print(f"Delta Score: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"Delta Hit Rate: {'+' if delta_hit_rate >= 0 else ''}{delta_hit_rate:.2f}")
    
    # Release Gate logic
    # Approved if:
    # 1. Delta is positive or zero (score did not degrade)
    # 2. V2 Score is at least 3.5
    # 3. Retrieval metrics in V2 are healthy (Hit Rate >= 0.8, MRR >= 0.6)
    score_pass = delta >= 0 and v2_summary["metrics"]["avg_score"] >= 3.5
    retrieval_pass = v2_summary["metrics"]["hit_rate"] >= 0.8 and v2_summary["metrics"]["mrr"] >= 0.6
    
    decision = "APPROVE" if (score_pass and retrieval_pass) else "BLOCK RELEASE"
    
    # Add regression section to summary file for report richness
    v2_summary["regression"] = {
        "v1_metrics": v1_summary["metrics"],
        "delta_score": delta,
        "delta_hit_rate": delta_hit_rate,
        "delta_mrr": delta_mrr,
        "decision": decision
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if decision == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
