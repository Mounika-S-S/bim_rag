# src/evaluation/evaluator.py
"""
Complete evaluator for retrieval metrics only
FIXED: Better error handling and document ID matching
"""
import json
import os
import numpy as np
from typing import List, Dict, Any, Set
from datetime import datetime
from collections import defaultdict

from src.retrieval.query_router import QueryRouter
from src.evaluation.metrics import (
    compute_all_retrieval_metrics,
    compute_overall_metrics,
    compute_metrics_by_type
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Evaluator:
    """Complete evaluation system for retrieval metrics"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_path = f"data/processed/{project_id}"
        self.results_path = os.path.join(self.project_path, "evaluation", "results")
        
        self.router = QueryRouter(project_id)
        self.test_queries = self._load_test_queries()
        
        # Results storage
        self.per_query_results = []
        self.routing_results = []
    
    def _load_test_queries(self) -> List[Dict]:
        query_path = os.path.join(self.project_path, "evaluation", "test_queries.json")
        if os.path.exists(query_path):
            with open(query_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {data['total_queries']} test queries")
            return data['queries']
        print("❌ No test queries found")
        return []
    
    def _get_document_id(self, result: Dict) -> str:
        """Extract document ID from result metadata"""
        metadata = result.get('metadata', {})
        layer = metadata.get('layer', 'unknown')
        
        if layer == 'comparison':
            # Comparison records have id field
            return f"l45_{metadata.get('id', 'unknown')}"
        elif layer in ['L1', 'L2', 'L4', 'L5']:
            # Regular layers
            return f"{layer.lower()}_{metadata.get('id', 'unknown')}"
        elif layer == 'compliance':
            # Compliance issues
            return f"mis_{metadata.get('id', metadata.get('element_id', 'unknown'))}"
        else:
            return f"{layer}_{metadata.get('id', 'unknown')}"
    
    def run_evaluation(self, top_k: int = 10) -> Dict:
        """Run complete evaluation for retrieval metrics"""
        if not self.test_queries:
            return {}
        
        print(f"\n{'='*70}")
        print(f"🔍 RUNNING RETRIEVAL EVALUATION")
        print(f"{'='*70}")
        print(f"Project: {self.project_id}")
        print(f"Test queries: {len(self.test_queries)}")
        print(f"Top-k: {top_k}")
        print(f"{'='*70}\n")
        
        os.makedirs(self.results_path, exist_ok=True)
        
        for i, test in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}] {test['query_type']}: {test['query'][:60]}...")
            
            try:
                # Get retrieval results
                response = self.router.retrieve(test['query'], top_k=top_k)
                
                # Extract retrieved document IDs
                retrieved_ids = []
                relevance_scores = {}
                
                for result in response['results']:
                    doc_id = self._get_document_id(result)
                    retrieved_ids.append(doc_id)
                    
                    # Binary relevance (1 if relevant, 0 otherwise)
                    if doc_id in test.get('relevant_ids', []):
                        relevance_scores[doc_id] = 1.0
                    else:
                        relevance_scores[doc_id] = 0.0
                
                # Debug: Show what was retrieved vs what's relevant
                relevant_set = set(test.get('relevant_ids', []))
                if relevant_set:
                    matched = set(retrieved_ids) & relevant_set
                    print(f"      Retrieved: {len(retrieved_ids)} docs, Relevant: {len(relevant_set)} docs, Matched: {len(matched)}")
                    if matched:
                        print(f"      ✓ Matched IDs: {list(matched)[:3]}")
                
                # ===== RETRIEVAL METRICS =====
                retrieval_metrics = compute_all_retrieval_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_set,
                    relevance_scores=relevance_scores,
                    ks=[1, 3, 5, 10]
                )
                
                # Routing accuracy
                expected = test.get('expected_retriever', 'unknown')
                actual = response['route']['retriever']
                route_correct = (actual == expected)
                
                # Store results
                query_result = {
                    "query_id": test['id'],
                    "query": test['query'],
                    "query_type": test['query_type'],
                    "expected_retriever": expected,
                    "actual_retriever": actual,
                    "route_correct": route_correct,
                    "router_confidence": response['route']['confidence'],
                    "retrieved_count": len(retrieved_ids),
                    "relevant_count": len(relevant_set),
                }
                
                # Add retrieval metrics
                for key, value in retrieval_metrics.items():
                    query_result[key] = float(value) if isinstance(value, (np.floating, float)) else value
                
                self.per_query_results.append(query_result)
                self.routing_results.append({
                    "query_id": test['id'],
                    "query_type": test['query_type'],
                    "expected": expected,
                    "actual": actual,
                    "correct": route_correct,
                    "confidence": response['route']['confidence']
                })
                
                # Print summary
                status = "✅" if route_correct else "❌"
                print(f"   Router: {actual} (expected: {expected}) - {status}")
                print(f"   Hit@5: {retrieval_metrics['hit_rate@5']:.2f}, MRR: {retrieval_metrics['mrr']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute overall statistics
        overall_stats = self._compute_overall_stats()
        
        # Save all results
        self._save_results(overall_stats)
        self._print_summary(overall_stats)
        
        return overall_stats
    
    def _compute_overall_stats(self) -> Dict:
        """Compute overall statistics for all metrics"""
        if not self.per_query_results:
            return {}
        
        # Compute overall metrics
        overall = compute_overall_metrics(self.per_query_results)
        
        # Compute metrics by type
        by_type = compute_metrics_by_type(self.per_query_results)
        
        # Routing accuracy
        correct_routes = sum(1 for r in self.routing_results if r['correct'])
        routing_acc = correct_routes / len(self.routing_results) if self.routing_results else 0
        
        # Add routing accuracy to overall
        overall['routing_accuracy'] = float(routing_acc)
        
        return {
            "total_queries": len(self.per_query_results),
            "overall": overall,
            "by_type": by_type
        }
    
    def _save_results(self, overall_stats: Dict):
        """Save all evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overall metrics
        overall_file = os.path.join(self.results_path, f"overall_metrics_{timestamp}.json")
        with open(overall_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "project": self.project_id,
                "stats": overall_stats
            }, f, indent=2, cls=NumpyEncoder)
        
        # Save per-query results
        per_query_file = os.path.join(self.results_path, "per_query_results.json")
        with open(per_query_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.per_query_results
            }, f, indent=2, cls=NumpyEncoder)
        
        # Save routing results
        routing_file = os.path.join(self.results_path, "routing_results.json")
        with open(routing_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.routing_results
            }, f, indent=2, cls=NumpyEncoder)
        
        # Save latest for dashboard
        latest_file = os.path.join(self.results_path, "latest_results.json")
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "stats": overall_stats
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✅ Results saved to {overall_file}")
    
    def _print_summary(self, overall_stats: Dict):
        """Print comprehensive summary"""
        print(f"\n{'='*70}")
        print(f"📊 RETRIEVAL EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Queries: {overall_stats['total_queries']}")
        print(f"Routing Accuracy: {overall_stats['overall']['routing_accuracy']:.1%}")
        
        print(f"\n📈 OVERALL RETRIEVAL METRICS")
        print(f"{'-'*50}")
        overall = overall_stats['overall']
        print(f"MAP:               {overall.get('map', 0):.3f}")
        print(f"MRR:               {overall.get('avg_mrr', 0):.3f}")
        print(f"Hit Rate@1:        {overall.get('avg_hit_rate@1', 0):.1%}")
        print(f"Hit Rate@3:        {overall.get('avg_hit_rate@3', 0):.1%}")
        print(f"Hit Rate@5:        {overall.get('avg_hit_rate@5', 0):.1%}")
        print(f"NDCG@5:            {overall.get('avg_ndcg@5', 0):.1%}")
        print(f"Precision@5:       {overall.get('avg_precision@5', 0):.1%}")
        print(f"Recall@5:          {overall.get('avg_recall@5', 0):.1%}")
        print(f"F1@5:              {overall.get('avg_f1@5', 0):.1%}")
        
        print(f"\n📊 BY QUERY TYPE")
        print(f"{'-'*70}")
        print(f"{'Type':10s} {'Count':6s} {'Hit@5':8s} {'MRR':8s} {'NDCG@5':8s} {'F1@5':8s}")
        print(f"{'-'*70}")
        
        for qtype, stats in sorted(overall_stats['by_type'].items()):
            print(f"{qtype:10s} {stats['query_count']:6d} "
                  f"{stats.get('avg_hit_rate@5', 0):7.1%} "
                  f"{stats.get('avg_mrr', 0):8.3f} "
                  f"{stats.get('avg_ndcg@5', 0):7.1%} "
                  f"{stats.get('avg_f1@5', 0):7.1%}")


if __name__ == "__main__":
    evaluator = Evaluator("new")
    evaluator.run_evaluation()