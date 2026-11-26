"""Create/update the LangSmith gold dataset and evaluate the live Mini RAG API.

Usage:
    pip install -r requirements.txt
    python evaluate_langsmith.py

Requires LANGSMITH_API_KEY, LANGSMITH_TRACING=true, and a running API with
the gold document already ingested. The script never uploads document contents.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from langsmith import Client

# Load project .env values with the same Pydantic settings used by the API.
from app.config import settings
if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_TRACING", str(settings.langsmith_tracing).lower())
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)

ROOT = Path(__file__).resolve().parent
DATASET_NAME = os.getenv("LANGSMITH_DATASET", "mini-rag-gold-qa")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
GOLD_PATH = Path(os.getenv("GOLD_QA_PATH", str(ROOT / "gold_qa.json")))


def load_cases() -> list[dict[str, Any]]:
    with GOLD_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def ensure_dataset(client: Client, cases: list[dict[str, Any]]) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Using existing LangSmith dataset: {DATASET_NAME}")
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Mini RAG gold questions for answer-term coverage and source citation checks.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": item["question"]},
                "outputs": {
                    "expected_terms": item["expected_terms"],
                    "expected_source": item["expected_source"],
                },
                "metadata": {"benchmark": "mini-rag-gold-qa"},
            }
            for item in cases
        ],
    )
    print(f"Created LangSmith dataset: {DATASET_NAME} ({len(cases)} examples)")


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/api/query",
        json={"question": inputs["question"], "top_k": 12},
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "timings_ms": result.get("timings_ms", {}),
        "usage": result.get("usage", {}),
    }


def answer_term_coverage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    answer = str(outputs.get("answer", "")).casefold()
    terms = reference_outputs.get("expected_terms", [])
    matched = sum(str(term).casefold() in answer for term in terms)
    score = matched / max(1, len(terms))
    return {"key": "answer_term_coverage", "score": score, "comment": f"Matched {matched}/{len(terms)} expected terms."}


def expected_source_cited(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = str(reference_outputs.get("expected_source", "")).casefold()
    citations = outputs.get("citations", [])
    found = any(expected in str(c.get("source", "")).casefold() for c in citations)
    return {"key": "expected_source_cited", "score": float(found), "comment": f"Expected source: {expected or 'not provided'}"}


def has_citations(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    citations = outputs.get("citations", [])
    return {"key": "has_citations", "score": float(bool(citations)), "comment": f"Returned {len(citations)} citation(s)."}


def latency_under_5s(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    total_ms = float(outputs.get("timings_ms", {}).get("total", 1e12))
    return {"key": "latency_under_5s", "score": float(total_ms <= 5000), "comment": f"Observed end-to-end latency: {total_ms:.1f} ms."}


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY in .env before running this evaluation.")
    cases = load_cases()
    client = Client()
    ensure_dataset(client, cases)
    response = requests.get(f"{API_URL}/health", timeout=15)
    response.raise_for_status()
    print(f"API health: {response.json()}")
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[answer_term_coverage, expected_source_cited, has_citations, latency_under_5s],
        experiment_prefix="mini-rag-baseline",
        description="Baseline evaluation of the Mini RAG API against the gold QA set.",
        max_concurrency=1,
    )
    print("Evaluation submitted to LangSmith.")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Experiment: {getattr(results, 'experiment_name', 'see LangSmith dashboard')}")
    print("Open LangSmith → Datasets / Experiments to inspect per-example feedback and aggregate scores.")


if __name__ == "__main__":
    main()
