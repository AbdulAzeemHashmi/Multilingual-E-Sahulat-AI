"""
Comprehensive Quantitative Benchmark and Evaluation Suite
Computes:
- Recall@1, Recall@5, MRR@5, MAP@5
- Per-language breakdown (English, Urdu Script, Roman Urdu)
- Abstention Precision, Recall, and F1
- Citation Support Rate
- Cross-Script Consistency
- Retrieval Latency (ms)
Compares: BM25 vs Dense vs Hybrid RRF
Exports validation pack JSON, CSV, and summary tables.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.retriever import MultilingualRetriever
from src.generator import GroundedAnswerEngine


class MultilingualEvaluator:
    """
    Evaluates Multilingual E-Sahulat AI on benchmark queries.
    """

    def __init__(
        self,
        retriever: Optional[MultilingualRetriever] = None,
        generator: Optional[GroundedAnswerEngine] = None
    ):
        self.retriever = retriever or MultilingualRetriever()
        self.generator = generator or GroundedAnswerEngine()

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        methods: List[str] = ['bm25', 'dense', 'hybrid'],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation over dataset for all specified retrieval methods.
        """
        evaluation_results = {}
        all_query_predictions = {m: [] for m in methods}

        for method in methods:
            print(f"\nEvaluating method: {method.upper()} on {len(dataset)} queries...")
            metrics_overall = self._evaluate_single_method(dataset, method, top_k)
            evaluation_results[method] = metrics_overall

        return evaluation_results

    def _evaluate_single_method(
        self,
        dataset: List[Dict[str, Any]],
        method: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Evaluates a single retrieval method."""
        answerable_queries = [r for r in dataset if r.get('answerable')]
        unanswerable_queries = [r for r in dataset if not r.get('answerable')]

        reciprocal_ranks = []
        hits_at_1 = []
        hits_at_5 = []
        latencies = []

        # Breakdown by language
        lang_metrics = defaultdict(lambda: {'rr': [], 'h1': [], 'h5': [], 'count': 0})

        # Evaluate Answerable Queries for Retrieval Performance
        predictions_log = []
        for q_record in answerable_queries:
            query = q_record['query']
            gold_intent = q_record['intent_id']
            lang = q_record.get('language', 'en')

            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(query, method=method, top_k=top_k)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            retrieved_ids = [p['passage_id'] for p in retrieved]

            # Calculate metrics
            h1 = 1 if retrieved_ids and retrieved_ids[0] == gold_intent else 0
            h5 = 1 if gold_intent in retrieved_ids[:5] else 0

            if gold_intent in retrieved_ids:
                rank = retrieved_ids.index(gold_intent) + 1
                rr = 1.0 / rank
            else:
                rr = 0.0

            hits_at_1.append(h1)
            hits_at_5.append(h5)
            reciprocal_ranks.append(rr)

            lang_metrics[lang]['h1'].append(h1)
            lang_metrics[lang]['h5'].append(h5)
            lang_metrics[lang]['rr'].append(rr)
            lang_metrics[lang]['count'] += 1

            # Answer generation
            ans_res = self.generator.synthesize_answer(query, retrieved, target_lang=lang)

            predictions_log.append({
                "qid": q_record.get('qid'),
                "query": query,
                "language": lang,
                "intent_id": gold_intent,
                "answerable": True,
                "retrieved_top1": retrieved_ids[0] if retrieved_ids else None,
                "retrieved_top5": retrieved_ids,
                "hit@1": bool(h1),
                "hit@5": bool(h5),
                "reciprocal_rank": rr,
                "status": ans_res['status'],
                "cited_passage": ans_res.get('passage_id'),
                "answer": ans_res.get('answer'),
                "latency_ms": round(latency, 2)
            })

        # Evaluate Abstention on Unanswerable Queries
        abstention_correct = 0
        abstention_false_answers = 0
        for q_record in unanswerable_queries:
            query = q_record['query']
            lang = q_record.get('language', 'en')
            topic = q_record.get('topic', 'unanswerable')

            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(query, method=method, top_k=top_k)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            ans_res = self.generator.synthesize_answer(query, retrieved, target_lang=lang)
            is_abstained = (ans_res['status'] == 'abstained')

            if is_abstained:
                abstention_correct += 1
            else:
                abstention_false_answers += 1

            predictions_log.append({
                "qid": q_record.get('qid'),
                "query": query,
                "language": lang,
                "intent_id": None,
                "topic": topic,
                "answerable": False,
                "retrieved_top1": retrieved[0]['passage_id'] if retrieved else None,
                "retrieved_top5": [p['passage_id'] for p in retrieved],
                "hit@1": False,
                "hit@5": False,
                "reciprocal_rank": 0.0,
                "status": ans_res['status'],
                "abstention_reason": ans_res.get('abstention_reason'),
                "answer": ans_res.get('answer'),
                "latency_ms": round(latency, 2)
            })

        # Calculate Overall Metrics
        total_ans = len(answerable_queries)
        total_unans = len(unanswerable_queries)
        recall_at_1 = sum(hits_at_1) / total_ans if total_ans else 0.0
        recall_at_5 = sum(hits_at_5) / total_ans if total_ans else 0.0
        mrr_at_5 = sum(reciprocal_ranks) / total_ans if total_ans else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Abstention Metrics
        abstention_recall = abstention_correct / total_unans if total_unans else 1.0
        # Check false abstentions on answerable queries
        false_abstentions = sum(1 for p in predictions_log if p['answerable'] and p['status'] == 'abstained')
        total_abstentions_claimed = abstention_correct + false_abstentions
        abstention_precision = abstention_correct / total_abstentions_claimed if total_abstentions_claimed else 1.0
        abstention_f1 = (2 * abstention_precision * abstention_recall) / (abstention_precision + abstention_recall) if (abstention_precision + abstention_recall) else 0.0

        # Citation support rate: answered queries whose cited passage equals the gold intent
        correct_citations = sum(1 for p in predictions_log if p['answerable'] and p['cited_passage'] == p['intent_id'])
        citation_support_rate = correct_citations / total_ans if total_ans else 0.0

        # Compute Cross-Script Consistency (Triplet analysis)
        intent_triplets = defaultdict(dict)
        for p in predictions_log:
            if p['answerable'] and p['intent_id']:
                intent_triplets[p['intent_id']][p['language']] = p['hit@5']

        consistent_intents = 0
        evaluable_triplets = 0
        for iid, lang_hits in intent_triplets.items():
            if 'en' in lang_hits and 'ur' in lang_hits and 'roman_ur' in lang_hits:
                evaluable_triplets += 1
                if lang_hits['en'] and lang_hits['ur'] and lang_hits['roman_ur']:
                    consistent_intents += 1

        cross_script_consistency = consistent_intents / evaluable_triplets if evaluable_triplets else 1.0

        # Format Language Breakdown
        by_language = {}
        for l, d in lang_metrics.items():
            cnt = d['count']
            by_language[l] = {
                "count": cnt,
                "recall@1": round(sum(d['h1']) / cnt, 4) if cnt else 0.0,
                "recall@5": round(sum(d['h5']) / cnt, 4) if cnt else 0.0,
                "mrr@5": round(sum(d['rr']) / cnt, 4) if cnt else 0.0
            }

        return {
            "method": method,
            "sample_size": len(dataset),
            "answerable_count": total_ans,
            "unanswerable_count": total_unans,
            "recall@1": round(recall_at_1, 4),
            "recall@5": round(recall_at_5, 4),
            "mrr@5": round(mrr_at_5, 4),
            "citation_support_rate": round(citation_support_rate, 4),
            "abstention_precision": round(abstention_precision, 4),
            "abstention_recall": round(abstention_recall, 4),
            "abstention_f1": round(abstention_f1, 4),
            "cross_script_consistency": round(cross_script_consistency, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "by_language": by_language,
            "predictions": predictions_log
        }


def run_full_evaluation(output_dir: str = 'validation_pack'):
    """Runs evaluation across test set (60) and full dataset (150) and saves outputs."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    with open('data/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open('data/train_dev.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    full_data = train_data + test_data

    evaluator = MultilingualEvaluator()

    # 1. Run on Test Set (Primary Evaluation)
    print("==================================================")
    print("RUNNING BENCHMARK EVALUATION ON OFFICIAL TEST SET (N=60)")
    print("==================================================")
    test_eval = evaluator.evaluate_dataset(test_data, methods=['bm25', 'dense', 'hybrid'], top_k=5)

    # 2. Run on Full Dataset (N=150)
    print("\n==================================================")
    print("RUNNING BENCHMARK EVALUATION ON FULL DATASET (N=150)")
    print("==================================================")
    full_eval = evaluator.evaluate_dataset(full_data, methods=['bm25', 'dense', 'hybrid'], top_k=5)

    # Build Comparison DataFrame for Test Set
    comp_rows = []
    for m in ['bm25', 'dense', 'hybrid']:
        res = test_eval[m]
        comp_rows.append({
            "Method": m.upper(),
            "Recall@1": f"{res['recall@1']*100:.1f}%",
            "Recall@5": f"{res['recall@5']*100:.1f}%",
            "MRR@5": f"{res['mrr@5']:.4f}",
            "Citation Support Rate": f"{res['citation_support_rate']*100:.1f}%",
            "Abstention F1": f"{res['abstention_f1']*100:.1f}%",
            "Cross-Script Consistency": f"{res['cross_script_consistency']*100:.1f}%",
            "Avg Latency (ms)": f"{res['avg_latency_ms']:.1f} ms"
        })

    df_comp = pd.DataFrame(comp_rows)
    comp_csv_path = os.path.join(output_dir, 'retrieval_comparison.csv')
    df_comp.to_csv(comp_csv_path, index=False)

    print("\n=== RETRIEVAL COMPARISON TABLE (TEST SET N=60) ===")
    print(df_comp.to_string(index=False))

    # Save detailed test predictions from Hybrid method
    test_predictions = test_eval['hybrid']['predictions']
    pred_json_path = os.path.join(output_dir, 'test_predictions.json')
    pred_csv_path = os.path.join(output_dir, 'test_predictions.csv')

    with open(pred_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_predictions, f, indent=2, ensure_ascii=False)

    df_preds = pd.DataFrame(test_predictions)
    df_preds.to_csv(pred_csv_path, index=False, encoding='utf-8')

    # Save complete evaluation metrics
    eval_json_path = os.path.join(output_dir, 'full_evaluation_metrics.json')
    # Remove predictions object from summary metrics to keep clean
    summary_eval = {
        "test_split_metrics": {m: {k: v for k, v in test_eval[m].items() if k != 'predictions'} for m in test_eval},
        "full_dataset_metrics": {m: {k: v for k, v in full_eval[m].items() if k != 'predictions'} for m in full_eval}
    }
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_eval, f, indent=2, ensure_ascii=False)

    # Save markdown table for reports
    headers = list(df_comp.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df_comp.iterrows():
        md_lines.append("| " + " | ".join(str(val) for val in row) + " |")
    md_table = "\n".join(md_lines)

    with open(os.path.join(output_dir, 'retrieval_comparison.md'), 'w', encoding='utf-8') as f:
        f.write("# Retrieval Comparison Table\n\n" + md_table + "\n")

    print(f"\nArtifacts saved successfully in {output_dir}/")
    return test_eval, full_eval


if __name__ == '__main__':
    run_full_evaluation()
