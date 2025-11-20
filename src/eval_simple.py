import os
import csv
import json
from time import perf_counter_ns
from typing import Dict, Any, List

import numpy as np

from .config import DATA_DIR
from .rag_llm import answer

EVALS_DIR = os.path.join(DATA_DIR, "evals")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(EVALS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET = os.path.join(EVALS_DIR, "dataset.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "eval_report.json")

# seuil de réussite d'une question : au moins 70% des faits présents
FACT_HIT_THRESHOLD = 0.6


def parse_facts(row: Dict[str, str]) -> List[str]:
    """
    Récupère la liste des 'expected_facts' à partir de la ligne CSV.
    - Si 'expected_facts' existe: split sur '|'
    - Sinon, si 'expected_substring' existe: on le considère comme un seul "fait"
    """
    if "expected_facts" in row and row["expected_facts"].strip():
        raw = row["expected_facts"]
        parts = [p.strip() for p in raw.split("|")]
        return [p for p in parts if p]
    elif "expected_substring" in row and row["expected_substring"].strip():
        return [row["expected_substring"].strip()]
    else:
        return []


def eval_row(row: Dict[str, str]) -> Dict[str, Any]:
    q = row["question"].strip()
    k = int(row.get("k", 3))

    facts = parse_facts(row)
    facts_lower = [f.lower() for f in facts]

    t0 = perf_counter_ns()
    ans = answer(q, k=k)
    dt_ms = (perf_counter_ns() - t0) / 1e6

    ans_lower = ans.lower()

    fact_hits = 0
    for f in facts_lower:
        if f and f in ans_lower:
            fact_hits += 1

    nb_facts = len(facts_lower)
    fact_ratio = (fact_hits / nb_facts) if nb_facts > 0 else 0.0

    # question considérée "OK" si au moins FACT_HIT_THRESHOLD des faits sont présents
    hit_question = int(fact_ratio >= FACT_HIT_THRESHOLD)

    return {
        "id": row.get("id", ""),
        "question": q,
        "k": k,
        "facts": facts,
        "fact_hits": fact_hits,
        "fact_ratio": fact_ratio,
        "hit_question": hit_question,
        "latency_ms": dt_ms,
        "answer": ans,
    }


def main():
    if not os.path.exists(DATASET):
        raise SystemExit(f"{DATASET} introuvable. Crée le CSV d'évaluation.")

    with open(DATASET, newline="", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("Dataset vide.")

    results = [eval_row(r) for r in rows]

    n = len(results)
    hits_questions = sum(r["hit_question"] for r in results)
    recall_questions = hits_questions / n

    # couverture moyenne des faits sur toutes les questions
    avg_fact_ratio = float(np.mean([r["fact_ratio"] for r in results])) if results else 0.0

    latencies = [r["latency_ms"] for r in results]
    latencies_sorted = sorted(latencies)
    p50 = float(np.percentile(latencies_sorted, 50))
    p95 = float(np.percentile(latencies_sorted, 95))

    report = {
        "n": n,
        "recall_questions": recall_questions,
        "avg_fact_ratio": avg_fact_ratio,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "fact_hit_threshold": FACT_HIT_THRESHOLD,
        "results": results,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    recall_pct = round(recall_questions * 100, 1)
    avg_fact_pct = round(avg_fact_ratio * 100, 1)

    summary = {
        "n": n,
        "recall_questions": round(recall_questions, 3),
        "recall_questions_pct": f"{recall_pct}%",
        "avg_fact_ratio": round(avg_fact_ratio, 3),
        "avg_fact_ratio_pct": f"{avg_fact_pct}%",
        "p50_ms": int(p50),
        "p95_ms": int(p95),
        "threshold_fact_ok": FACT_HIT_THRESHOLD,
        "out": OUT_PATH,
    }

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
