"""
RAG Evaluation Script
Runs the golden dataset against the RAG pipeline and measures:
  - Faithfulness (answer supported by context)
  - Answer correctness (semantic similarity to ground truth)
  - Context relevance (retrieved chunks relevant to query)

Uses RAGAS for structured evaluation, with a lightweight fallback if RAGAS
is not installed (cosine similarity only).

Usage:
    python evaluation/evaluate.py [--debug] [--fail-on-threshold]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import evaluation_config
from pipeline import RAGPipeline
from utils.logger import logger, setup_logger


# ── Evaluation helpers ────────────────────────────────────────────────────────

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_answer_correctness(
    predicted: str,
    ground_truth: str,
    embedder,
) -> float:
    """
    Compute semantic similarity between predicted answer and ground truth
    using embedding cosine similarity.
    """
    emb_pred = embedder.embed_query(predicted)
    emb_gt = embedder.embed_query(ground_truth)
    return cosine_similarity(emb_pred, emb_gt)


def compute_context_relevance(
    query: str,
    retrieved_texts: List[str],
    embedder,
) -> float:
    """
    Average cosine similarity between the query and each retrieved chunk.
    """
    if not retrieved_texts:
        return 0.0
    query_emb = embedder.embed_query(query)
    chunk_embs = embedder.embed_texts(retrieved_texts)
    sims = [cosine_similarity(query_emb, ce) for ce in chunk_embs]
    return sum(sims) / len(sims)


def compute_faithfulness(answer: str, context_texts: List[str]) -> float:
    """
    Heuristic faithfulness: fraction of answer sentences that contain
    at least one token overlap with the context.

    RAGAS provides a proper LLM-based faithfulness metric; this is
    the offline fallback.
    """
    if not context_texts:
        return 0.0

    context_blob = " ".join(context_texts).lower()
    context_tokens = set(context_blob.split())

    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        tokens = set(sentence.lower().split())
        overlap = tokens & context_tokens
        if len(overlap) / max(len(tokens), 1) > 0.2:  # 20% token overlap
            supported += 1

    return supported / len(sentences)


# ── RAGAS evaluator ───────────────────────────────────────────────────────────

def try_ragas_eval(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Attempt RAGAS evaluation. Returns empty dict if RAGAS is unavailable.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            context_relevancy,
            faithfulness,
        )

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, context_relevancy, answer_correctness],
        )
        return dict(result)

    except ImportError:
        logger.warning(
            "RAGAS not installed — using lightweight metric fallback. "
            "Install with: pip install ragas datasets"
        )
        return {}
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {}


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(debug: bool = False) -> Dict[str, Any]:
    """
    Run evaluation against the golden dataset.

    Returns:
        Dict with aggregate scores and per-question results.
    """
    setup_logger(debug=debug)
    logger.info("=== Starting RAG Evaluation ===")

    # Load golden dataset
    dataset_path = Path(evaluation_config.golden_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at {dataset_path}. "
            "Please create evaluation/golden_dataset.json first."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    logger.info(f"Loaded {len(golden)} evaluation examples")

    # Initialize pipeline (without LLM to save API costs during eval)
    pipeline = RAGPipeline(debug=debug)

    questions, answers, contexts, ground_truths = [], [], [], []
    per_question_results = []

    for item in golden:
        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"Evaluating [{qid}]: {question[:60]}…")
        start = time.perf_counter()

        try:
            response = pipeline.query(question)
            elapsed = time.perf_counter() - start

            retrieved_texts = [rc.chunk.text for rc in response.retrieved_chunks]

            # Lightweight metrics (always available)
            answer_correctness = compute_answer_correctness(
                response.answer, ground_truth, pipeline._embedder
            )
            context_relevance = compute_context_relevance(
                question, retrieved_texts, pipeline._embedder
            )
            faithfulness = compute_faithfulness(response.answer, retrieved_texts)

            result = {
                "id": qid,
                "question": question,
                "predicted_answer": response.answer,
                "ground_truth": ground_truth,
                "is_fallback": response.is_fallback,
                "answer_correctness": round(answer_correctness, 4),
                "context_relevance": round(context_relevance, 4),
                "faithfulness": round(faithfulness, 4),
                "num_chunks_retrieved": len(response.retrieved_chunks),
                "latency_seconds": round(elapsed, 3),
                "passed": (
                    answer_correctness >= evaluation_config.answer_correctness_threshold
                    and context_relevance >= evaluation_config.context_relevance_threshold
                    and faithfulness >= evaluation_config.faithfulness_threshold
                ),
            }
            logger.info(
                f"  [{qid}] correctness={answer_correctness:.3f} "
                f"relevance={context_relevance:.3f} "
                f"faithfulness={faithfulness:.3f} "
                f"passed={result['passed']}"
            )

        except Exception as e:
            logger.error(f"Evaluation failed for [{qid}]: {e}")
            result = {
                "id": qid,
                "question": question,
                "error": str(e),
                "passed": False,
            }

        per_question_results.append(result)
        questions.append(question)
        answers.append(result.get("predicted_answer", ""))
        contexts.append([rc.chunk.text for rc in response.retrieved_chunks] if "is_fallback" in result else [])
        ground_truths.append(ground_truth)

    # Aggregate
    passed = [r for r in per_question_results if r.get("passed")]
    failed = [r for r in per_question_results if not r.get("passed")]

    def avg(key):
        vals = [r[key] for r in per_question_results if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregate = {
        "total": len(golden),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / len(golden), 4),
        "avg_answer_correctness": avg("answer_correctness"),
        "avg_context_relevance": avg("context_relevance"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_latency_seconds": avg("latency_seconds"),
    }

    # Try RAGAS (may be richer)
    ragas_scores = try_ragas_eval(questions, answers, contexts, ground_truths)
    if ragas_scores:
        aggregate["ragas"] = ragas_scores

    # Save results
    output_dir = Path(evaluation_config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "aggregate": aggregate,
        "per_question": per_question_results,
        "failed_cases": [r for r in per_question_results if not r.get("passed")],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_file}")
    logger.info(f"=== RESULTS: pass_rate={aggregate['pass_rate']} | "
                f"correctness={aggregate['avg_answer_correctness']} | "
                f"relevance={aggregate['avg_context_relevance']} | "
                f"faithfulness={aggregate['avg_faithfulness']} ===")

    return report


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System Evaluator")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with code 1 if scores fall below thresholds (CI/CD mode)",
    )
    args = parser.parse_args()

    report = run_evaluation(debug=args.debug)
    agg = report["aggregate"]

    # Print summary table
    print("\n" + "=" * 60)
    print("  RAG EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions : {agg['total']}")
    print(f"  Passed          : {agg['passed']}")
    print(f"  Failed          : {agg['failed']}")
    print(f"  Pass rate       : {agg['pass_rate']:.1%}")
    print(f"  Avg correctness : {agg['avg_answer_correctness']:.4f}")
    print(f"  Avg relevance   : {agg['avg_context_relevance']:.4f}")
    print(f"  Avg faithfulness: {agg['avg_faithfulness']:.4f}")
    print(f"  Avg latency     : {agg['avg_latency_seconds']}s")
    print("=" * 60)

    if args.fail_on_threshold or evaluation_config.fail_build_on_threshold:
        thresholds_met = (
            agg["avg_faithfulness"] >= evaluation_config.faithfulness_threshold
            and agg["avg_answer_correctness"] >= evaluation_config.answer_correctness_threshold
            and agg["avg_context_relevance"] >= evaluation_config.context_relevance_threshold
        )
        if not thresholds_met:
            print("\n❌  EVALUATION FAILED — scores below thresholds. Build blocked.")
            sys.exit(1)
        print("\n✅  All evaluation thresholds passed.")
