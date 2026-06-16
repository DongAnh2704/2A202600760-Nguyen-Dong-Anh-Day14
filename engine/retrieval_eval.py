from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Calculates if at least one expected_id is in the top_k of retrieved_ids.
        """
        if not expected_ids:
            # For adversarial or out-of-context, no documents are expected.
            # If the agent also retrieved nothing, we can count it as a hit, but usually
            # we just skip these cases when calculating retrieval metrics.
            # Here we return 1.0 if both are empty, else 0.0, but in evaluate_batch we skip empty expected_ids.
            return 1.0 if not retrieved_ids else 0.0
            
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR).
        Finds the first position of an expected_id in retrieved_ids (1-indexed).
        MRR = 1 / position. Returns 0.0 if not found.
        """
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
            
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_case(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> Dict[str, float]:
        """
        Evaluate a single case and return Hit Rate and MRR.
        """
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids)
        }

    async def evaluate_batch(self, dataset: List[Dict], retrieved_dataset: List[List[str]]) -> Dict:
        """
        Runs evaluation on a batch.
        """
        total_hit_rate = 0.0
        total_mrr = 0.0
        count = 0
        
        for case, retrieved_ids in zip(dataset, retrieved_dataset):
            expected_ids = case.get("expected_retrieval_ids", [])
            # Only count cases where we actually expect retrieval (i.e. not out-of-context or adversarial)
            if expected_ids:
                metrics = self.evaluate_case(expected_ids, retrieved_ids)
                total_hit_rate += metrics["hit_rate"]
                total_mrr += metrics["mrr"]
                count += 1
                
        avg_hit_rate = total_hit_rate / count if count > 0 else 1.0
        avg_mrr = total_mrr / count if count > 0 else 1.0
        
        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "evaluated_cases": count
        }
