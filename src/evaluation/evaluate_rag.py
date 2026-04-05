"""
evaluate_rag_only.py - Complete RAG Evaluation WITHOUT any LLM
Only RAG metrics (retrieval + semantic quality) - No LLM evaluation metrics
All metrics computed using deterministic methods.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

# For embedding similarity (Answer Relevance)
from sentence_transformers import SentenceTransformer


class RAGEvaluator:
    """
    Complete RAG Evaluation WITHOUT any LLM calls.
    Evaluates both retrieval quality and semantic quality.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with embedding model for similarity calculations.
        This model is small (80MB) and runs locally.
        """
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model_name)
        print("✓ Embedding model loaded")
    
    # =========================================================
    # PART 1: DETERMINISTIC RETRIEVAL METRICS (TIER 1)
    # =========================================================
    
    def compute_hit_rate(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """
        Hit Rate@K: Does any relevant document appear in top-K?
        Formula: 1 if any relevant doc in top-K, else 0
        """
        top_k_ids = retrieved_ids[:k]
        return 1.0 if any(rid in top_k_ids for rid in relevant_ids) else 0.0
    
    def compute_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank: 1 / rank of first relevant document
        Formula: 1 / position_of_first_relevant_doc
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    def compute_ndcg(self, retrieved_ids: List[str], relevance_scores: Dict[str, int], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        relevance_scores: 0=irrelevant, 1=relevant, 2=highly relevant
        """
        # Get relevance scores for retrieved documents
        y_true = [relevance_scores.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
        
        if len(y_true) == 0 or sum(y_true) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = sum(y_true[i] / np.log2(i + 2) for i in range(len(y_true)))
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(ideal_scores[i] / np.log2(i + 2) for i in range(len(ideal_scores)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_precision_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """
        Precision@K: (Relevant docs in top-K) / K
        Formula: (#relevant_in_topK) / K
        """
        top_k_ids = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for rid in top_k_ids if rid in relevant_ids)
        return relevant_in_top_k / k
    
    def compute_recall_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
        """
        Recall@K: (Relevant docs in top-K) / (Total relevant docs)
        Formula: (#relevant_in_topK) / (#total_relevant)
        """
        top_k_ids = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for rid in top_k_ids if rid in relevant_ids)
        total_relevant = len(relevant_ids)
        return relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
    
    # =========================================================
    # PART 2: DETERMINISTIC SEMANTIC RAG METRICS (TIER 2)
    # =========================================================
    
    def compute_context_relevance_keyword(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Context Relevance using keyword overlap
        Formula: (unique keywords in query that appear in retrieved docs) / (unique keywords in query)
        """
        # Stopwords to filter out common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'shall', 'should', 'may', 'might', 'must', 'can', 'could',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'for', 'and', 'nor', 'but', 'or', 'yet', 'so', 'of', 'to', 'in',
            'for', 'on', 'by', 'with', 'without', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under'
        }
        
        # Extract keywords from query
        query_tokens = set([
            w.lower() for w in query.split() 
            if w.lower() not in stopwords and len(w) > 2
        ])
        
        if not query_tokens:
            return 1.0
        
        # Combine all retrieved docs
        all_doc_text = " ".join(retrieved_docs).lower()
        
        # Count how many query tokens appear in docs
        matched_tokens = sum(1 for token in query_tokens if token in all_doc_text)
        
        return matched_tokens / len(query_tokens)
    
    def compute_faithfulness_ngram(self, answer: str, retrieved_docs: List[str], n: int = 3) -> float:
        """
        Faithfulness using n-gram overlap
        Formula: (n-grams in answer that appear in context) / (total n-grams in answer)
        """
        # Combine all retrieved docs
        context = " ".join(retrieved_docs).lower()
        answer_lower = answer.lower()
        
        # Generate n-grams from answer
        answer_words = answer_lower.split()
        
        if len(answer_words) < n:
            n = max(1, len(answer_words))
        
        if len(answer_words) < n:
            return 1.0
        
        answer_ngrams = set()
        for i in range(len(answer_words) - n + 1):
            ngram = " ".join(answer_words[i:i+n])
            answer_ngrams.add(ngram)
        
        if not answer_ngrams:
            return 1.0
        
        # Count supported n-grams
        supported = 0
        for ngram in answer_ngrams:
            if ngram in context:
                supported += 1
        
        return supported / len(answer_ngrams)
    
    def compute_answer_relevance_cosine(self, query: str, answer: str) -> float:
        """
        Answer Relevance using cosine similarity
        Formula: cosine_similarity(embedding(query), embedding(answer))
        """
        query_emb = self.embedder.encode(query)
        answer_emb = self.embedder.encode(answer)
        
        # Cosine similarity
        similarity = np.dot(query_emb, answer_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(answer_emb)
        )
        return float(similarity)
    
    def compute_context_recall_term(self, ground_truth_context: str, retrieved_docs: List[str]) -> float:
        """
        Context Recall using term overlap
        Formula: (unique terms in ground truth that appear in retrieved docs) / (unique terms in ground truth)
        """
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'shall', 'should', 'may', 'might', 'must', 'can', 'could'
        }
        
        # Extract terms from ground truth context
        gt_terms = set([
            w.lower() for w in ground_truth_context.split() 
            if w.lower() not in stopwords and len(w) > 2
        ])
        
        if not gt_terms:
            return 1.0
        
        # Combine all retrieved docs
        all_doc_text = " ".join(retrieved_docs).lower()
        
        # Count matched terms
        matched_terms = sum(1 for term in gt_terms if term in all_doc_text)
        
        return matched_terms / len(gt_terms)
    
    # =========================================================
    # PART 3: COMPLETE EVALUATION FOR A SINGLE QUERY
    # =========================================================
    
    def evaluate_query(
        self,
        query: str,
        retrieved_ids: List[str],
        retrieved_docs: List[str],
        relevant_ids: List[str],
        relevance_scores: Dict[str, int],
        generated_answer: str,
        ground_truth_context: str,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Complete evaluation for one query using ALL deterministic methods.
        
        Args:
            query: User's question
            retrieved_ids: List of document IDs retrieved (ordered by rank)
            retrieved_docs: List of document texts retrieved (ordered by rank)
            relevant_ids: List of ground truth relevant document IDs
            relevance_scores: Dict mapping doc_id -> relevance grade (0,1,2)
            generated_answer: Answer generated by LLM
            ground_truth_context: Expected context that should be retrieved
            k: Number of retrieved documents to consider
            
        Returns:
            Dictionary with all RAG metrics
        """
        results = {}
        
        # TIER 1: Deterministic Retrieval Metrics
        results["hit_rate"] = self.compute_hit_rate(retrieved_ids, relevant_ids, k)
        results["mrr"] = self.compute_mrr(retrieved_ids, relevant_ids)
        results["ndcg"] = self.compute_ndcg(retrieved_ids, relevance_scores, k)
        results["precision"] = self.compute_precision_at_k(retrieved_ids, relevant_ids, k)
        results["recall"] = self.compute_recall_at_k(retrieved_ids, relevant_ids, k)
        
        # TIER 2: Deterministic Semantic RAG Metrics
        results["context_relevance"] = self.compute_context_relevance_keyword(query, retrieved_docs)
        results["faithfulness"] = self.compute_faithfulness_ngram(generated_answer, retrieved_docs)
        results["answer_relevance"] = self.compute_answer_relevance_cosine(query, generated_answer)
        results["context_recall"] = self.compute_context_recall_term(ground_truth_context, retrieved_docs)
        
        return results
    
    # =========================================================
    # PART 4: BATCH EVALUATION
    # =========================================================
    
    def evaluate_batch(self, test_data: List[Dict], k: int = 5) -> Dict[str, Any]:
        """
        Evaluate multiple queries and return aggregated results.
        
        test_data format:
        [
            {
                "query": str,
                "retrieved_ids": List[str],
                "retrieved_docs": List[str],
                "relevant_ids": List[str],
                "relevance_scores": Dict[str, int],
                "generated_answer": str,
                "ground_truth_context": str
            },
            ...
        ]
        """
        all_results = []
        
        for item in test_data:
            result = self.evaluate_query(
                query=item["query"],
                retrieved_ids=item["retrieved_ids"],
                retrieved_docs=item["retrieved_docs"],
                relevant_ids=item["relevant_ids"],
                relevance_scores=item["relevance_scores"],
                generated_answer=item["generated_answer"],
                ground_truth_context=item["ground_truth_context"],
                k=k
            )
            all_results.append(result)
        
        # Aggregate results
        aggregated = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "per_query": values
            }
        
        aggregated["num_queries"] = len(test_data)
        
        return aggregated
    
    # =========================================================
    # PART 5: GENERATE COMPARISON TABLE (Your vs Friend)
    # =========================================================
    
    def generate_comparison_table(self, your_results: Dict, friend_results: Dict) -> None:
        """
        Generate the three-column comparison table.
        
        your_results: Results from this evaluator (deterministic only)
        friend_results: Results from friend (using LLM/RAGAS)
        
        Note: LLM-required metrics (context_relevance, faithfulness, 
              answer_relevance, context_recall) will show "-" for friend if not available.
        """
        metrics = [
            "hit_rate", "mrr", "ndcg", "precision", "recall",
            "context_relevance", "faithfulness", "answer_relevance", "context_recall"
        ]
        
        print("\n" + "="*110)
        print("📊 RAG EVALUATION COMPARISON TABLE")
        print("="*110)
        print(f"\n{'Evaluation Metric':<25} {'Deterministic Only':>22} {'Semantic Only (LLM)':>22} {'Both (Combined)':>22}")
        print("-"*110)
        
        for metric in metrics:
            # Deterministic: Your results (no LLM)
            det_score = your_results.get(metric, {}).get("mean", 0.0) if isinstance(your_results.get(metric), dict) else your_results.get(metric, 0.0)
            
            # Semantic: Friend's results (LLM required) - if not available, show "-"
            if friend_results and metric in friend_results:
                sem_score = friend_results[metric] if isinstance(friend_results[metric], (int, float)) else friend_results[metric].get("mean", 0.0)
                sem_display = f"{sem_score:.4f}"
            else:
                sem_display = "-"
                sem_score = det_score  # For both column fallback
            
            # Both: For retrieval metrics = deterministic, for RAG metrics = semantic (or deterministic if semantic not available)
            if metric in ["hit_rate", "mrr", "ndcg", "precision", "recall"]:
                both_score = det_score
            else:
                both_score = sem_score if friend_results else det_score
            
            print(f"{metric:<25} {det_score:>22.4f} {sem_display:>22} {both_score:>22.4f}")
        
        print("="*110)
        print("\n📝 NOTE:")
        print("  - 'Deterministic Only': Computed using keyword/ID matching (No LLM)")
        print("  - 'Semantic Only (LLM)': Requires LLM (RAGAS framework)")
        print("  - 'Both (Combined)': Retrieval metrics = Deterministic, RAG metrics = Semantic")
        print("  - '-' indicates LLM-required metric not available in friend's results")
        print("="*110)


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    print("="*70)
    print("🚀 RAG EVALUATION - NO LLM REQUIRED")
    print("="*70)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # =========================================================
    # EXAMPLE TEST DATA (Replace with your actual data)
    # =========================================================
    
    test_data = [
        {
            "query": "What are fire resistance requirements for external walls?",
            "retrieved_ids": ["L4_fire_001", "L5_8559daca", "L4_fire_002", "L1_wall_001", "L2_product_001"],
            "retrieved_docs": [
                "External walls must have minimum fire resistance of 120 minutes.",
                "REQ-FIRE-001 requires all external walls to achieve EI120.",
                "Fire doors require 60 minutes fire resistance.",
                "Wall YC-ST-WA-EIP has fire rating 120 minutes.",
                "Product X has fire rating 90 minutes."
            ],
            "relevant_ids": ["L4_fire_001", "L5_8559daca"],
            "relevance_scores": {
                "L4_fire_001": 2,
                "L5_8559daca": 2,
                "L4_fire_002": 0,
                "L1_wall_001": 1,
                "L2_product_001": 0
            },
            "generated_answer": "External walls need 120 minutes fire resistance according to regulations.",
            "ground_truth_context": "External walls must have minimum fire resistance of 120 minutes."
        },
        {
            "query": "What is REQ-FIRE-001?",
            "retrieved_ids": ["L5_8559daca", "L4_fire_001", "L5_requirement_002", "L1_wall_002", "L2_product_002"],
            "retrieved_docs": [
                "REQ-FIRE-001 requires all external walls to achieve EI120.",
                "External walls must have minimum fire resistance of 120 minutes.",
                "REQ-STR-001 requires load capacity 5kN/m.",
                "Wall YC-ST-WA-EIP has fire rating 120 minutes.",
                "Product Y has fire rating 120 minutes."
            ],
            "relevant_ids": ["L5_8559daca"],
            "relevance_scores": {
                "L5_8559daca": 2,
                "L4_fire_001": 1,
                "L5_requirement_002": 0,
                "L1_wall_002": 0,
                "L2_product_002": 0
            },
            "generated_answer": "REQ-FIRE-001 requires all external walls to achieve EI120 fire resistance.",
            "ground_truth_context": "REQ-FIRE-001 requires all external walls to achieve EI120 fire resistance."
        },
        {
            "query": "Show me non-compliant external walls",
            "retrieved_ids": ["L124_mismatch_001", "L5_mismatch_001", "L1_wall_003", "L4_fire_001", "L2_product_003"],
            "retrieved_docs": [
                "Wall YC-AR-WA-IIP is non-compliant due to missing fire rating.",
                "External wall YC-AR-WA-IIP missing fire rating.",
                "Wall YC-ST-WA-003 has fire rating 60 minutes.",
                "External walls require 120 minutes fire resistance.",
                "Product Z has fire rating 90 minutes."
            ],
            "relevant_ids": ["L124_mismatch_001", "L5_mismatch_001"],
            "relevance_scores": {
                "L124_mismatch_001": 2,
                "L5_mismatch_001": 2,
                "L1_wall_003": 1,
                "L4_fire_001": 0,
                "L2_product_003": 0
            },
            "generated_answer": "Wall YC-AR-WA-IIP is non-compliant due to missing fire rating.",
            "ground_truth_context": "Wall YC-AR-WA-IIP is non-compliant due to missing fire rating."
        }
    ]
    
    # Run evaluation
    results = evaluator.evaluate_batch(test_data, k=5)
    
    # Print detailed results
    print("\n" + "="*70)
    print("📊 DETAILED EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*65)
    
    for metric in ["hit_rate", "mrr", "ndcg", "precision", "recall", 
                   "context_relevance", "faithfulness", "answer_relevance", "context_recall"]:
        if metric in results:
            print(f"{metric:<25} {results[metric]['mean']:>10.4f} {results[metric]['std']:>10.4f} "
                  f"{results[metric]['min']:>10.4f} {results[metric]['max']:>10.4f}")
    
    print("="*70)
    print(f"\n✅ Evaluation completed for {results['num_queries']} queries")
    
    # =========================================================
    # GENERATE COMPARISON TABLE (Your results vs Friend's results)
    # =========================================================
    
    # For friend's results (LLM-required metrics would be here)
    # Since friend doesn't have results yet, we use None to show "-"
    friend_results = None
    
    # If friend has results, uncomment and fill:
    # friend_results = {
    #     "context_relevance": 0.91,
    #     "faithfulness": 0.95,
    #     "answer_relevance": 0.93,
    #     "context_recall": 0.85
    # }
    
    evaluator.generate_comparison_table(results, friend_results)