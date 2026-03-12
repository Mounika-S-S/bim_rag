# src/retrieval/l4_retriever.py
"""
Retriever for Layer 4 (Regulations) only
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore


class L4Retriever(BaseRetriever):
    """
    Handles queries about regulations only
    Examples: "What does NBC say about fire?", "Fire code requirements"
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L4Retriever"
        self.l4_data = JSONStorage.load(project_id, "L4_regulation.json") or []
        self.vector_store = UnifiedVectorStore()
        
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve regulations matching query
        Primarily semantic search due to text nature
        """
        # Regulations are best found via semantic search
        semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.8)
        
        results = []
        for res in semantic_results:
            if res['metadata'].get('layer') == 'L4':
                results.append(self.format_result(
                    content=res['content'],
                    source="L4_regulation",
                    layer=4,
                    score=res['relevance_score'],
                    metadata=res['metadata']
                ))
        
        return results[:top_k]