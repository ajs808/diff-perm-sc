__all__ = ['ndcg_at_k', 'mrr', 'evaluate_retrieval']

from typing import Dict, List
import numpy as np

from ..data import RankingExample


def dcg_at_k(scores: List[float], k: int) -> float:
    scores = scores[:k]
    if not scores:
        return 0.0
    
    gains = np.array(scores)
    positions = np.arange(1, len(gains) + 1)
    discounted_gains = gains / np.log2(positions + 1)
    return float(np.sum(discounted_gains))


def ndcg_at_k(scores: List[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k.
    
    Args:
        scores: List of relevance scores (0 or positive integers)
        k: Cutoff rank
        
    Returns:
        NDCG@k value between 0 and 1
    """
    if not scores:
        return 0.0
    
    dcg = dcg_at_k(scores, k)
    
    ideal_scores = sorted(scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr(scores: List[float]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        scores: List of relevance scores (0 or positive integers)
        
    Returns:
        Reciprocal rank of first relevant item, or 0 if none
    """
    for i, score in enumerate(scores):
        if score > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(results: Dict[str, RankingExample], 
                      qrels: Dict[str, Dict[str, int]], 
                      k_values: List[int] = [10, 100]) -> Dict[str, float]:
    """
    Evaluate retrieval results against relevance judgments.
    
    Args:
        results: Dictionary mapping query_id to RankingExample
        qrels: Dictionary mapping query_id to dict of passage_id -> relevance_score
        k_values: List of k values for NDCG@k computation
        
    Returns:
        Dictionary with metric names as keys and average scores as values
    """
    ndcg_scores = {f'ndcg@{k}': [] for k in k_values}
    mrr_scores = []
    
    for query_id, ranking_example in results.items():
        query_qrels = qrels.get(query_id, {})
        
        if not query_qrels:
            continue
        
        relevance_scores = []
        for hit in ranking_example.hits:
            passage_id = hit.id
            relevance = query_qrels.get(passage_id, 0)
            relevance_scores.append(relevance)
        
        for k in k_values:
            ndcg = ndcg_at_k(relevance_scores, k)
            ndcg_scores[f'ndcg@{k}'].append(ndcg)
        
        mrr_score = mrr(relevance_scores)
        mrr_scores.append(mrr_score)
    
    metrics = {}
    for k in k_values:
        metric_name = f'ndcg@{k}'
        if ndcg_scores[metric_name]:
            metrics[metric_name] = np.mean(ndcg_scores[metric_name])
        else:
            metrics[metric_name] = 0.0
    
    if mrr_scores:
        metrics['mrr'] = np.mean(mrr_scores)
    else:
        metrics['mrr'] = 0.0
    
    return metrics

