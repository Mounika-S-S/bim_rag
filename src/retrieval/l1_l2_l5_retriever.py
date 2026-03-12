# src/retrieval/l1_l2_l5_retriever.py
"""
Retriever for L1+L2+L5 compliance queries using schema-based mismatch data
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore
import logging

logger = logging.getLogger(__name__)


class L1L2L5Retriever(BaseRetriever):
    """
    Handles project compliance queries (L1+L2+L5)
    Uses schema-validated mismatch_l5.json from compliance_engine_l5.py
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L1L2L5Retriever"
        
        # Load L5-specific mismatch data (from compliance_engine_l5)
        self.mismatch_l5 = JSONStorage.load(project_id, "mismatch_l5.json") or []
        
        # Also load main mismatch for any L5-tagged issues
        self.mismatch_all = JSONStorage.load(project_id, "mismatch.json") or []
        self.l1_l2_l5_issues = [
            issue for issue in self.mismatch_all 
            if issue.get('layer_check') == 'L1_L2_L5'
        ]
        
        # Combine both sources
        self.all_issues = self.mismatch_l5 + self.l1_l2_l5_issues
        
        # Load vector store for semantic fallback
        self.vector_store = UnifiedVectorStore()
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
            logger.info(f"✅ Loaded vector store with {len(self.vector_store.text_chunks)} chunks")
        
        logger.info(f"📊 Loaded {len(self.all_issues)} L1+L2+L5 compliance issues")
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve L1+L2+L5 compliance issues
        """
        query_lower = query.lower()
        results = []
        seen_elements = set()  # Avoid duplicates
        
        # ============ STEP 1: Direct schema-based mismatch lookup ============
        # This uses the structured data from compliance_engine_l5.py
        for issue in self.all_issues:
            # Create unique key to avoid duplicates
            element_id = issue.get('element_id', '')
            req_id = issue.get('requirement_id', '')
            unique_key = f"{element_id}_{req_id}"
            
            if unique_key in seen_elements:
                continue
            seen_elements.add(unique_key)
            
            # Extract fields from schema-validated data
            element_name = issue.get('element_name', '').lower()
            req_id = issue.get('requirement_id', '').lower()
            rule = issue.get('rule', '').lower()
            issue_text = issue.get('issue', '').lower()
            priority = issue.get('priority', 'Medium')
            
            # Calculate relevance score
            score = 0.0
            
            # Exact element name match (highest)
            if element_name and element_name in query_lower:
                score += 0.9
            
            # Requirement ID match
            if req_id and req_id in query_lower:
                score += 0.8
            
            # Priority match (high/medium/low)
            if priority.lower() in query_lower:
                score += 0.4
            
            # Rule text match
            if rule and any(word in rule for word in query_lower.split()):
                score += 0.3
            
            # Issue description match
            if issue_text and any(word in issue_text for word in query_lower.split()):
                score += 0.2
            
            if score > 0:
                # Create rich content from schema data
                status = "NON-COMPLIANT"
                content = (f"[PROJECT COMPLIANCE] {issue.get('element_name', 'Unknown')} "
                          f"is {status}: {issue.get('issue', '')} "
                          f"[Priority: {issue.get('priority', 'Medium')}]")
                
                results.append(self.format_result(
                    content=content,
                    source="mismatch_l5",
                    layer="compliance",
                    score=min(score, 1.0),
                    metadata={
                        "element_id": issue.get('element_id'),
                        "element_name": issue.get('element_name'),
                        "element_type": issue.get('element_type'),
                        "requirement_id": issue.get('requirement_id'),
                        "requirement_text": issue.get('requirement_text'),
                        "rule": issue.get('rule'),
                        "actual_value": issue.get('actual_value'),
                        "required_value": issue.get('required_value'),
                        "priority": issue.get('priority'),
                        "layer_check": "L1_L2_L5"
                    }
                ))
        
        # ============ STEP 2: Semantic search fallback ============
        if len(results) < top_k:
            semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.6)
            
            for res in semantic_results:
                # Prioritize L5 and compliance chunks
                if res['metadata'].get('layer') in ['L5', 'compliance']:
                    content_preview = res['content'][:50]
                    if not any(content_preview in r['content'] for r in results):
                        results.append(self.format_result(
                            content=res['content'],
                            source=res['metadata'].get('source', 'unknown'),
                            layer=res['metadata'].get('layer', 0),
                            score=res['relevance_score'] * 0.8,
                            metadata=res['metadata']
                        ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]