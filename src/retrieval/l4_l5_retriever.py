# src/retrieval/l4_l5_retriever.py
"""
Retriever for L4-L5 compliance queries
Handles questions comparing regulations with project requirements
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore
from src.utils.element_utils import extract_numbers, contains_dimension_query
import logging

logger = logging.getLogger(__name__)


class L4L5Retriever(BaseRetriever):
    """
    Handles queries comparing regulations (L4) with requirements (L5)
    Examples:
    - "Compare rules with regulations"
    - "Is project stricter than code?"
    - "What are the gaps between rules and code?"
    - "Show me where project exceeds code"
    - "Which requirements are missing from project?"
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L4L5Retriever"
        
        # Load L4-L5 comparison data
        self.l4_l5_comparisons = JSONStorage.load(project_id, "l4_l5_mismatch.json") or []
        
        # Load raw data for context
        self.l4 = JSONStorage.load(project_id, "L4_regulation.json") or []
        self.l5 = JSONStorage.load(project_id, "L5_requirement.json") or []
        
        # Load vector store for semantic fallback
        self.vector_store = UnifiedVectorStore()
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
            logger.info(f"✅ Loaded vector store with {len(self.vector_store.text_chunks)} chunks")
        
        logger.info(f"📊 Loaded {len(self.l4_l5_comparisons)} L4-L5 comparisons")
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve L4-L5 comparisons matching query
        """
        query_lower = query.lower()
        results = []
        seen = set()
        
        # ============ STEP 1: Direct comparison lookup ============
        # Identify query intent
        is_comparison = any(word in query_lower for word in [
            'compare', 'vs', 'versus', 'difference', 'gap',
            'stricter', 'weaker', 'stronger', 'exceed'
        ])
        
        is_gap = any(word in query_lower for word in [
            'gap', 'missing', 'lack', 'absent', 'not covered'
        ])
        
        is_stricter = any(word in query_lower for word in [
            'stricter', 'exceed', 'more stringent', 'higher', 'tighter'
        ])
        
        # Check for topic-specific queries
        topics = {
            'fire': ['fire', 'flame', 'burn', 'combustible'],
            'height': ['height', 'tall', 'storey', 'floor', 'storeys'],
            'thickness': ['thickness', 'thin', 'thick'],
            'width': ['width', 'wide', 'narrow', 'corridor'],
            'parking': ['parking', 'car', 'vehicle', 'garage'],
            'setback': ['setback', 'distance', 'front', 'side', 'rear'],
            'area': ['area', 'square', 'sqm', 'floor area']
        }
        
        detected_topics = []
        for topic, keywords in topics.items():
            if any(k in query_lower for k in keywords):
                detected_topics.append(topic)
        
        # Search comparisons
        for comp in self.l4_l5_comparisons:
            comp_id = comp.get('comparison_id', '')
            if comp_id in seen:
                continue
            seen.add(comp_id)
            
            l4_rule = comp.get('l4_rule', {})
            l5_req = comp.get('l5_requirement', {})
            relationship = comp.get('relationship', '')
            gap_desc = comp.get('gap_description', '').lower()
            l4_text = l4_rule.get('text', '').lower() if l4_rule else ''
            l5_text = l5_req.get('text', '').lower() if l5_req else ''
            
            score = 0.0
            
            # Topic match
            if detected_topics:
                for topic in detected_topics:
                    if topic in l4_text or topic in l5_text:
                        score += 0.5
                        break
            
            # Intent match
            if is_comparison and 'comparison' in comp.get('comparison', ''):
                score += 0.3
            
            if is_gap and relationship == 'missing':
                score += 0.8
            
            if is_stricter and relationship == 'stricter':
                score += 0.7
            
            # Number extraction for numeric queries
            if contains_dimension_query(query):
                numbers = extract_numbers(query)
                l4_val = l4_rule.get('value') if l4_rule else None
                l5_val = l5_req.get('value') if l5_req else None
                
                if numbers and l4_val:
                    try:
                        if abs(float(l4_val) - numbers[0]) / numbers[0] < 0.2:  # 20% tolerance
                            score += 0.6
                    except:
                        pass
                
                if numbers and l5_val:
                    try:
                        if abs(float(l5_val) - numbers[0]) / numbers[0] < 0.2:
                            score += 0.6
                    except:
                        pass
            
            # Specific relationship queries
            if 'stricter' in query_lower and relationship == 'stricter':
                score += 0.9
            if 'weaker' in query_lower and relationship == 'weaker':
                score += 0.9
            if 'equal' in query_lower and relationship == 'equal':
                score += 0.9
            if 'missing' in query_lower and relationship == 'missing':
                score += 0.9
            
            if score > 0:
                # Format based on relationship
                if relationship == 'missing':
                    icon = "❌"
                    content = f"{icon} [GAP] {comp.get('gap_description', 'Missing requirement')}"
                elif relationship == 'stricter':
                    icon = "🔼"
                    content = f"{icon} [STRICTER] {comp.get('gap_description', 'Project exceeds code')}"
                elif relationship == 'weaker':
                    icon = "🔽"
                    content = f"{icon} [WEAKER] {comp.get('gap_description', 'Project below code')}"
                elif relationship == 'equal':
                    icon = "✅"
                    content = f"{icon} [EQUAL] {comp.get('gap_description', 'Matches code')}"
                elif relationship == 'extra':
                    icon = "✨"
                    content = f"{icon} [EXTRA] {comp.get('gap_description', 'Additional project requirement')}"
                else:
                    icon = "⚠️"
                    content = f"{icon} [COMPARISON] {comp.get('comparison', '')}"
                
                results.append(self.format_result(
                    content=content,
                    source="l4_l5_comparison",
                    layer="comparison",
                    score=min(score, 1.0),
                    metadata=comp
                ))
        
        # ============ STEP 2: Semantic search fallback ============
        if len(results) < top_k:
            semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.6)
            
            for res in semantic_results:
                if res['metadata'].get('layer') in ['L4', 'L5', 'comparison']:
                    content_preview = res['content'][:50]
                    if not any(content_preview in r['content'] for r in results):
                        results.append(self.format_result(
                            content=res['content'],
                            source=res['metadata'].get('source', 'unknown'),
                            layer=res['metadata'].get('layer', 0),
                            score=res['relevance_score'] * 0.7,
                            metadata=res['metadata']
                        ))
        
        # Sort by score and return
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]