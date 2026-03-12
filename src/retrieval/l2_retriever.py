# src/retrieval/l2_retriever.py
"""
Retriever for Layer 2 (Products) only
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore


class L2Retriever(BaseRetriever):
    """
    Handles queries about products only
    Examples: "Show me products with EI120", "List Siniat products"
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L2Retriever"
        self.l2_data = JSONStorage.load(project_id, "L2_product.json") or []
        self.vector_store = UnifiedVectorStore()
        
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve products matching query
        Prioritizes exact matches, falls back to semantic
        """
        query_lower = query.lower()
        results = []
        
        # Step 1: Exact matches on product name/manufacturer
        for product in self.l2_data:
            props = product.get('properties', {})
            name = props.get('Product_Name', '').lower()
            mfr = props.get('Manufacturer', '').lower()
            
            score = 0
            # High score for exact name match
            if name and name in query_lower:
                score += 0.8
            # Medium score for manufacturer match
            if mfr and mfr in query_lower:
                score += 0.5
            
            if score > 0:
                # Add key specs to content
                content = f"[PRODUCT] {props.get('Product_Name', 'Unknown')} from {props.get('Manufacturer', 'Unknown')}"
                if 'Fire_Rating_Hours' in props:
                    content += f" | Fire Rating: {props['Fire_Rating_Hours']}h"
                if 'Unit_Cost_INR' in props:
                    content += f" | Cost: ₹{props['Unit_Cost_INR']}"
                
                results.append(self.format_result(
                    content=content,
                    source="L2_product",
                    layer=2,
                    score=score,
                    metadata={
                        "id": product.get('id'),
                        "name": props.get('Product_Name'),
                        "manufacturer": props.get('Manufacturer'),
                        "fire_rating": props.get('Fire_Rating_Hours'),
                        "cost": props.get('Unit_Cost_INR')
                    }
                ))
        
        # Step 2: Semantic search for product descriptions
        if len(results) < top_k:
            semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.5)
            for res in semantic_results:
                if res['metadata'].get('layer') == 'L2':
                    results.append(self.format_result(
                        content=res['content'],
                        source="L2_product",
                        layer=2,
                        score=res['relevance_score'] * 0.7,
                        metadata=res['metadata']
                    ))
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]