# src/retrieval/l1_l2_l4_retriever.py
"""
Retriever for L1+L2+L4 compliance queries using schema-based mismatch data
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore
import logging

logger = logging.getLogger(__name__)


class L1L2L4Retriever(BaseRetriever):
    """
    Handles regulatory compliance queries (L1+L2+L4)
    Uses schema-validated mismatch.json from compliance_engine.py
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L1L2L4Retriever"
        
        # Load schema-validated mismatch data from compliance engine
        self.mismatch = JSONStorage.load(project_id, "mismatch.json") or []
        
        # Filter to L1+L2+L4 issues only
        self.l1_l2_l4_issues = [
            issue for issue in self.mismatch 
            if issue.get('layer_check') == 'L1_L2_L4'
        ]
        
        # Load vector store for semantic fallback
        self.vector_store = UnifiedVectorStore()
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
            logger.info(f"✅ Loaded vector store with {len(self.vector_store.text_chunks)} chunks")
        
        logger.info(f"📊 Loaded {len(self.l1_l2_l4_issues)} L1+L2+L4 compliance issues")
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve L1+L2+L4 compliance issues
        """
        query_lower = query.lower()
        results = []
        
        # ============ STEP 1: Direct schema-based mismatch lookup ============
        # This uses the structured data from compliance_engine.py
        for issue in self.l1_l2_l4_issues:
            # Extract fields from schema-validated data
            element_name = issue.get('element_name', '').lower()
            element_type = issue.get('element_type', '').lower()
            rule_text = issue.get('rule_text', '').lower()
            issue_text = issue.get('issue', '').lower()
            
            # Calculate relevance score based on query matching
            score = 0.0
            
            # Exact element name match (highest score)
            if element_name and element_name in query_lower:
                score += 0.8
            
            # Element type match
            if element_type and element_type in query_lower:
                score += 0.4
            
            # Rule text match
            if rule_text and any(word in rule_text for word in query_lower.split()):
                score += 0.3
            
            # Issue description match
            if issue_text and any(word in issue_text for word in query_lower.split()):
                score += 0.2
            
            if score > 0:
                # Create human-readable content from schema data
                content = (f"[REGULATORY COMPLIANCE] {issue.get('element_name', 'Unknown')} "
                          f"is NON-COMPLIANT: {issue.get('issue', '')}")
                
                results.append(self.format_result(
                    content=content,
                    source="mismatch",
                    layer="compliance",
                    score=min(score, 1.0),  # Cap at 1.0
                    metadata={
                        "element_id": issue.get('element_id'),
                        "element_name": issue.get('element_name'),
                        "element_type": issue.get('element_type'),
                        "rule_text": issue.get('rule_text'),
                        "required": issue.get('required'),
                        "actual": issue.get('product_value'),
                        "unit": issue.get('unit'),
                        "layer_check": "L1_L2_L4"
                    }
                ))
        
        # ============ STEP 2: Semantic search fallback ============
        if len(results) < top_k:
            semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.6)
            
            for res in semantic_results:
                # Only include relevant layers
                if res['metadata'].get('layer') in ['L1', 'L4', 'compliance']:
                    # Check if not already in results (avoid duplicates)
                    content_preview = res['content'][:50]
                    if not any(content_preview in r['content'] for r in results):
                        results.append(self.format_result(
                            content=res['content'],
                            source=res['metadata'].get('source', 'unknown'),
                            layer=res['metadata'].get('layer', 0),
                            score=res['relevance_score'] * 0.7,  # Lower weight for semantic
                            metadata=res['metadata']
                        ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]