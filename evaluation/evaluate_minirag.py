
import os
import csv
import json
import time
import re
import requests

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

# =========================================================
# CONFIGURATION
# =========================================================

API_URL = os.getenv(
    "MINIRAG_API_URL",
    "http://localhost:8000/api/query"
)

TOP_K = int(os.getenv("MINIRAG_TOP_K", "5"))

# Set this to the exact indexed filename if you want
# to evaluate only the C programming PDF.
SOURCE = os.getenv("MINIRAG_SOURCE") or None

OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMEOUT = 120

# =========================================================
# DATASET
# =========================================================

DATASET = [
    {
        "id": "Q1",
        "question": "What is the purpose of malloc() in C?",
        "reference_answer": (
            "malloc dynamically allocates memory at runtime "
            "and returns a pointer to the allocated memory. "
            "The allocated memory is uninitialized."
        ),
        "source_supported": True
    },
    {
        "id": "Q2",
        "question": (
            "Explain the five IPv4 address classes: "
            "A, B, C, D, and E."
        ),
        "reference_answer": None,
        "source_supported": False
    },
    {
        "id": "Q3",
        "question": "What are the seven layers of the OSI model?",
        "reference_answer": None,
        "source_supported": False
    },
    {
        "id": "Q4",
        "question": "What is the difference between TCP and UDP?",
        "reference_answer": None,
        "source_supported": False
    },
    {
        "id": "Q5",
        "question": "What is a pointer?",
        "reference_answer": (
            "A pointer is a variable that stores the memory "
            "address of another variable and can be dereferenced "
            "to access its value."
        ),
        "source_supported": True
    }
]

# =========================================================
# METRICS
# =========================================================

def tokenize(text):
    return re.findall(
        r"\b[a-z0-9_]+\b",
        str(text or "").lower()
    )


def calculate_token_metrics(reference, prediction):
    ref = tokenize(reference)
    pred = tokenize(prediction)

    if not ref or not pred:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }

    overlap = sum(
        (Counter(ref) & Counter(pred)).values()
    )

    precision = overlap / len(pred)
    recall = overlap / len(ref)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def calculate_mrr(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return round(1 / rank, 4)

    return 0.0


def calculate_retrieval_metrics(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None

    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    hits = retrieved & relevant

    precision = len(hits) / max(1, len(retrieved))
    recall = len(hits) / len(relevant)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": calculate_mrr(retrieved_ids, relevant)
    }


# =========================================================
# API
# =========================================================

def call_minirag(question):

    payload = {
        "question": question,
        "top_k": TOP_K
    }

    if SOURCE:
        payload["source"] = SOURCE

    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json=payload,
        timeout=TIMEOUT
    )

    client_latency_ms = (
        time.perf_counter() - start
    ) * 1000

    response.raise_for_status()

    return response.json(), client_latency_ms


# =========================================================
# EVALUATE ONE QUESTION
# =========================================================

def evaluate(item):

    print(f"\n{'=' * 65}")
    print(f"{item['id']}: {item['question']}")

    result = {
        "id": item["id"],
        "question": item["question"],
        "source_supported": item["source_supported"],
        "status": "failed",
        "answer": None,
        "client_latency_ms": None,
        "server_timings_ms": {},
        "retrieved_chunk_count": 0,
        "retrieved_chunks": [],
        "citation_count": 0,
        "answer_metrics": None,
        "retrieval_metrics": None,
        "usage": {},
        "error": None
    }

    try:
        response, client_latency = call_minirag(
            item["question"]
        )

        answer = response.get("answer", "")
        citations = response.get("citations", [])
        timings = response.get("timings_ms", {})
        usage = response.get("usage", {})

        # These IDs are Qdrant point IDs, as returned
        # by your current RAGService.query() method.
        retrieved_ids = [
            str(citation["id"])
            for citation in citations
            if citation.get("id") is not None
        ]

        result.update({
            "status": "success",
            "answer": answer,
            "client_latency_ms": round(client_latency, 2),
            "server_timings_ms": timings,
            "retrieved_chunk_count": len(citations),
            "retrieved_chunks": citations,
            "citation_count": len(citations),
            "usage": usage
        })

        # Answer overlap: only for questions with
        # a source-grounded reference answer.
        if item["reference_answer"] and answer:
            result["answer_metrics"] = (
                calculate_token_metrics(
                    item["reference_answer"],
                    answer
                )
            )

        # Retrieval metrics need manually labelled
        # relevant Qdrant point IDs.
        relevant_ids = item.get("relevant_chunk_ids", [])

        if relevant_ids:
            result["retrieval_metrics"] = (
                calculate_retrieval_metrics(
                    retrieved_ids,
                    relevant_ids
                )
            )

        print(f"Answer: {answer}")
        print(f"Client latency: {client_latency:.2f} ms")
        print(f"Server timings: {timings}")
        print(f"Citations: {len(citations)}")
        print(f"Reranker: {usage.get('reranker')}")
        print(f"Tokens: {usage.get('total_tokens')}")
        print(f"Estimated cost: {usage.get('estimated_cost_usd')}")

        if result["answer_metrics"]:
            print(
                "Answer metrics:",
                result["answer_metrics"]
            )

    except requests.exceptions.RequestException as exc:
        result["error"] = str(exc)
        print(f"API ERROR: {exc}")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"EVALUATION ERROR: {exc}")

    return result


# =========================================================
# SUMMARY
# =========================================================

def average(values):
    return round(sum(values) / len(values), 4) if values else None


def build_summary(results):

    successful = [
        r for r in results
        if r["status"] == "success"
    ]

    latencies = [
        r["client_latency_ms"]
        for r in successful
        if r["client_latency_ms"] is not None
    ]

    retrieval_times = [
        r["server_timings_ms"]["retrieval"]
        for r in successful
        if "retrieval" in r["server_timings_ms"]
    ]

    reranking_times = [
        r["server_timings_ms"]["reranking"]
        for r in successful
        if "reranking" in r["server_timings_ms"]
    ]

    generation_times = [
        r["server_timings_ms"]["generation"]
        for r in successful
        if "generation" in r["server_timings_ms"]
    ]

    total_times = [
        r["server_timings_ms"]["total"]
        for r in successful
        if "total" in r["server_timings_ms"]
    ]

    scored = [
        r["answer_metrics"]
        for r in successful
        if r["answer_metrics"] is not None
    ]

    return {
        "total_questions": len(results),
        "successful_queries": len(successful),
        "failed_queries": len(results) - len(successful),

        "average_client_latency_ms": average(latencies),
        "average_server_total_ms": average(total_times),
        "average_retrieval_ms": average(retrieval_times),
        "average_reranking_ms": average(reranking_times),
        "average_generation_ms": average(generation_times),

        "average_answer_precision": average(
            [m["precision"] for m in scored]
        ),
        "average_answer_recall": average(
            [m["recall"] for m in scored]
        ),
        "average_answer_f1": average(
            [m["f1"] for m in scored]
        ),

        "answer_metric_type": "Token overlap baseline",
        "retrieval_metrics_status": (
            "Not evaluated: relevant chunk IDs not labelled."
        )
    }


# =========================================================
# EXPORT REPORTS
# =========================================================

def export_reports(results, summary):

    report = {
        "project": "MiniRAG",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": API_URL,
        "top_k": TOP_K,
        "source": SOURCE,
        "summary": summary,
        "results": results
    }

    json_path = OUTPUT_DIR / "evaluation_report.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    csv_path = OUTPUT_DIR / "evaluation_report.csv"

    fields = [
        "id",
        "question",
        "status",
        "source_supported",
        "client_latency_ms",
        "retrieved_chunk_count",
        "citation_count",
        "answer",
        "answer_metrics",
        "error"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for result in results:
            row = {
                key: result.get(key)
                for key in fields
            }

            row["answer_metrics"] = json.dumps(
                row["answer_metrics"]
            )

            writer.writerow(row)

    print("\nReports saved:")
    print(json_path)
    print(csv_path)


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 65)
    print("             MINIRAG EVALUATION")
    print("=" * 65)

    print("Endpoint:", API_URL)
    print("Top K:", TOP_K)
    print("Source:", SOURCE or "Automatic source routing")

    results = []

    for item in DATASET:
        results.append(evaluate(item))

    summary = build_summary(results)

    print("\n" + "=" * 65)
    print("                 FINAL SUMMARY")
    print("=" * 65)

    print(json.dumps(summary, indent=4))

    export_reports(results, summary)


if __name__ == "__main__":
    main()
