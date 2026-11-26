import json
import time
import statistics
from pathlib import Path

from evaluation.gold_dataset import GOLD_DATASET
from evaluation.metrics import calculate_precision_recall
from app.retrieval import retrieve_documents


TOP_K = 5


def calculate_ranking_metrics(retrieved, relevant_ids):

    ranked_ids = [
        chunk["chunk_id"]
        for chunk in retrieved
    ]

    relevant = set(relevant_ids)

    hits = [
        1 if chunk_id in relevant else 0
        for chunk_id in ranked_ids
    ]

    precision = sum(hits) / len(ranked_ids) if ranked_ids else 0
    recall = sum(hits) / len(relevant) if relevant else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    reciprocal_rank = 0

    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1 / rank
            break

    hit_rate = int(any(hits))

    dcg = sum(
        hit / __import__("math").log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )

    ideal_hits = min(len(relevant), TOP_K)

    idcg = sum(
        1 / __import__("math").log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg else 0

    return {
        "precision_at_5": precision,
        "recall_at_5": recall,
        "f1_at_5": f1,
        "mrr": reciprocal_rank,
        "hit_rate_at_5": hit_rate,
        "ndcg_at_5": ndcg
    }


def main():

    results = []

    for question, expected in GOLD_DATASET.items():

        start = time.perf_counter()

        chunks = retrieve_documents(
            question,
            top_k=TOP_K
        )

        latency = (time.perf_counter() - start) * 1000

        metrics = calculate_ranking_metrics(
            chunks,
            expected["relevant_chunk_ids"]
        )

        results.append({
            "question": question,
            "retrieved_chunks": len(chunks),
            "latency_ms": round(latency, 2),
            **metrics
        })

        print(f"\nQuestion: {question}")
        print(json.dumps(metrics, indent=2))

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "f1_at_5",
        "mrr",
        "hit_rate_at_5",
        "ndcg_at_5",
        "latency_ms"
    ]

    averages = {
        metric: round(
            statistics.mean(row[metric] for row in results),
            4
        )
        for metric in metric_names
    }

    output = {
        "total_questions": len(results),
        "top_k": TOP_K,
        "average_metrics": averages,
        "individual_results": results
    }

    Path("evaluation_results").mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        "evaluation_results/report.json",
        "w"
    ) as file:
        json.dump(output, file, indent=4)

    print("\n========== FINAL EVALUATION ==========")
    print(json.dumps(averages, indent=4))
    print("\nReport saved to evaluation_results/report.json")


if __name__ == "__main__":
    main()