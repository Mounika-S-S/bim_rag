# src/retrieval/l5_retriever.py
"""
Retriever for Layer 5 (Requirements) only
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore


class L5Retriever(BaseRetriever):
    """
    Handles queries about project requirements only
    Examples: "What are the project requirements?", "Show REQ-FIRE-001"
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L5Retriever"
        self.l5_data = JSONStorage.load(project_id, "L5_requirement.json") or []
        self.vector_store = UnifiedVectorStore()
        
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve requirements matching query
        Prioritizes exact RequirementID matches, then semantic
        """
        query_lower = query.lower()
        results = []
        
        # Step 1: Exact matches on RequirementID
        for req in self.l5_data:
            props = req.get('properties', {})
            req_id = props.get('RequirementID', '').lower()
            
            if req_id and req_id in query_lower:
                rule = f"{props.get('Property', '')} {props.get('Operator', '')} {props.get('RequiredValue', '')} {props.get('Unit', '')}"
                results.append(self.format_result(
                    content=f"[REQUIREMENT] {props.get('RequirementID')}: {props.get('Description', '')}",
                    source="L5_requirement",
                    layer=5,
                    score=1.0,
                    metadata={
                        "id": req.get('id'),
                        "requirement_id": props.get('RequirementID'),
                        "rule": rule,
                        "priority": props.get('Priority', 'Medium')
                    }
                ))
        
        # Step 2: Semantic search for requirement descriptions
        if len(results) < top_k:
            semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.8)
            for res in semantic_results:
                if res['metadata'].get('layer') == 'L5':
                    results.append(self.format_result(
                        content=res['content'],
                        source="L5_requirement",
                        layer=5,
                        score=res['relevance_score'],
                        metadata=res['metadata']
                    ))
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]