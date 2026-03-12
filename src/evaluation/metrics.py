# src/evaluation/metrics.py
"""
Core retrieval metrics for RAG systems
FIXED: Ensure all metric keys are consistent
"""
import numpy as np
import math
from typing import List, Dict, Set, Any
from collections import defaultdict


def precision_at_k(
    retrieved_ids: List[str], 
    relevant_ids: Set[str], 
    k: int = 5
) -> float:
    """Precision@K = |relevant ∩ retrieved| / k"""
    if k == 0 or not retrieved_ids:
        return 0.0
    
    top_k_ids = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_ids & relevant_ids)
    return float(relevant_retrieved / k)


def recall_at_k(
    retrieved_ids: List[str], 
    relevant_ids: Set[str], 
    k: int = 5
) -> float:
    """Recall@K = |relevant ∩ retrieved| / |relevant|"""
    if not relevant_ids:
        return 0.0
    
    top_k_ids = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_ids & relevant_ids)
    return float(relevant_retrieved / len(relevant_ids))


def f1_at_k(
    retrieved_ids: List[str], 
    relevant_ids: Set[str], 
    k: int = 5
) -> float:
    """F1@K = 2 * (P * R) / (P + R)"""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    
    if p + r == 0:
        return 0.0
    return float(2 * (p * r) / (p + r))


def mean_reciprocal_rank(
    retrieved_ids: List[str], 
    relevant_ids: Set[str]
) -> float:
    """MRR = 1 / rank of first relevant document"""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return float(1.0 / (i + 1))
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int = 5
) -> float:
    """nDCG@K = DCG@K / IDCG@K"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        gain = relevance_scores.get(doc_id, 0.0)
        dcg += gain / math.log2(i + 2)
    
    ideal_gains = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    
    return float(dcg / idcg if idcg > 0 else 0.0)


def hit_rate_at_k(
    retrieved_ids: List[str], 
    relevant_ids: Set[str], 
    k: int = 5
) -> float:
    """Hit Rate@K: Did any relevant document appear in top-k?"""
    if not relevant_ids:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    return float(1.0 if top_k_ids & relevant_ids else 0.0)


def average_precision(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """Average Precision (AP) for a single query"""
    if not relevant_ids:
        return 0.0
    
    relevant_found = 0
    precisions = []
    
    for k, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            relevant_found += 1
            precisions.append(relevant_found / k)
    
    return float(np.mean(precisions) if precisions else 0.0)


def compute_all_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    relevance_scores: Dict[str, float],
    ks: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Compute ALL retrieval metrics for a single query
    Returns dictionary with all metrics
    """
    metrics = {}
    
    # Core metrics
    metrics['mrr'] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    metrics['ap'] = average_precision(retrieved_ids, relevant_ids)
    
    # Metrics at different k values
    for k in ks:
        metrics[f'hit_rate@{k}'] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        metrics[f'precision@{k}'] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f'recall@{k}'] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f'f1@{k}'] = f1_at_k(retrieved_ids, relevant_ids, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved_ids, relevance_scores, k)
    
    return metrics


def compute_overall_metrics(
    per_query_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Aggregate per-query metrics into overall statistics
    """
    if not per_query_results:
        return {}
    
    aggregator = defaultdict(list)
    
    for result in per_query_results:
        for key, value in result.items():
            if isinstance(value, (int, float)) and not key.startswith('query_'):
                aggregator[key].append(float(value))
    
    overall = {
        "total_queries": len(per_query_results),
    }
    
    # Mean for each metric
    for metric, values in aggregator.items():
        overall[f'avg_{metric}'] = float(np.mean(values))
        overall[f'std_{metric}'] = float(np.std(values))
        overall[f'min_{metric}'] = float(np.min(values))
        overall[f'max_{metric}'] = float(np.max(values))
    
    # MAP (Mean Average Precision)
    overall['map'] = float(np.mean([r.get('ap', 0) for r in per_query_results]))
    
    return overall


def compute_metrics_by_type(
    per_query_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Group metrics by query type (L1, L2, L4, L5, L124, L125, L45)
    """
    by_type = defaultdict(list)
    
    for result in per_query_results:
        query_type = result.get('query_type', 'unknown')
        by_type[query_type].append(result)
    
    type_metrics = {}
    for query_type, results in by_type.items():
        type_metrics[query_type] = {
            "query_count": len(results),
            "avg_hit_rate@1": float(np.mean([r.get('hit_rate@1', 0) for r in results])),
            "avg_hit_rate@3": float(np.mean([r.get('hit_rate@3', 0) for r in results])),
            "avg_hit_rate@5": float(np.mean([r.get('hit_rate@5', 0) for r in results])),
            "avg_mrr": float(np.mean([r.get('mrr', 0) for r in results])),
            "avg_ndcg@5": float(np.mean([r.get('ndcg@5', 0) for r in results])),
            "avg_precision@5": float(np.mean([r.get('precision@5', 0) for r in results])),
            "avg_recall@5": float(np.mean([r.get('recall@5', 0) for r in results])),
            "avg_f1@5": float(np.mean([r.get('f1@5', 0) for r in results])),
            "map": float(np.mean([r.get('ap', 0) for r in results])),
        }
    
    return type_metrics