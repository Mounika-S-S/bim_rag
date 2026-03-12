# src/retrieval/l1_retriever.py
"""
Retriever for Layer 1 (IFC elements) only
Enhanced with entity-centric retrieval [SIGIR'23] and property indexing
"""
from src.retrieval.base_retriever import BaseRetriever
from src.core.json_storage import JSONStorage
from src.rag.unified_vector_store import UnifiedVectorStore
from src.utils.element_utils import extract_element_from_query
import logging
import re

logger = logging.getLogger(__name__)


class L1Retriever(BaseRetriever):
    """
    Handles queries about IFC elements only
    Examples: "Show me all walls", "List external walls"
    
    Based on entity-centric retrieval research [SIGIR'23]:
    - Entity name extraction and boosting
    - Property-level matching
    - Type-based filtering
    """
    
    def __init__(self, project_id):
        super().__init__(project_id)
        self.name = "L1Retriever"
        self.l1_data = JSONStorage.load(project_id, "L1_ifc.json") or []
        self.vector_store = UnifiedVectorStore()
        
        # Build property index for fast keyword matching
        self.property_index = self._build_property_index()
        
        # Load vector store if exists
        import os
        path = f"data/processed/{project_id}/unified.index"
        if os.path.exists(path):
            self.vector_store.load(path)
        
        logger.info(f"✅ L1Retriever initialized with {len(self.l1_data)} elements")
        logger.info(f"   Property index has {len(self.property_index)} entries")
    
    def _build_property_index(self):
        """Build inverted index of element properties for fast keyword search"""
        index = {}
        
        for element in self.l1_data:
            props = element.get('properties', {})
            element_id = element.get('id', 'unknown')
            
            # Index all property values
            for key, value in props.items():
                if value and isinstance(value, str):
                    # Split into words and index each
                    words = re.findall(r'\w+', value.lower())
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            if word not in index:
                                index[word] = []
                            index[word].append({
                                'element_id': element_id,
                                'element': element,
                                'property': key,
                                'value': value
                            })
        
        return index
    
    def _keyword_match_score(self, element, query_tokens):
        """Calculate keyword match score for an element"""
        props = element.get('properties', {})
        name = props.get('Name', '').lower()
        etype = element.get('entity_type', '').lower()
        
        score = 0
        matched_terms = set()
        
        for token in query_tokens:
            token_lower = token.lower()
            
            # Name matches (highest weight)
            if token_lower in name:
                score += 0.5
                matched_terms.add(token_lower)
            
            # Type matches
            if token_lower in etype:
                score += 0.3
                matched_terms.add(token_lower)
            
            # Property value matches
            for key, value in props.items():
                if value and isinstance(value, str):
                    if token_lower in value.lower():
                        score += 0.2
                        matched_terms.add(token_lower)
                        break
        
        # Bonus for matching multiple terms
        score *= (1 + 0.1 * len(matched_terms))
        
        return score
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve L1 elements matching query
        Multi-strategy retrieval [Microsoft RAG on Structured Data]
        """
        query_lower = query.lower()
        query_tokens = query_lower.split()
        results = []
        seen_ids = set()
        
        # === STRATEGY 1: Exact entity name match (highest confidence) ===
        element_name, confidence = extract_element_from_query(query)
        if element_name:
            logger.info(f"   Entity name detected: {element_name}")
            # Direct lookup in vector store by entity name
            entity_results = self.vector_store.search_by_entity_name(element_name, top_k)
            for res in entity_results:
                doc_id = res['metadata'].get('id')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(self.format_result(
                        content=res['content'],
                        source="L1_ifc",
                        layer=1,
                        score=res['relevance_score'] * 1.5,  # Boost exact matches
                        metadata=res['metadata']
                    ))
        
        # === STRATEGY 2: Keyword matching on all properties ===
        if len(results) < top_k * 2:
            for element in self.l1_data:
                element_id = element.get('id', 'unknown')
                if element_id in seen_ids:
                    continue
                
                score = self._keyword_match_score(element, query_tokens)
                
                if score > 0.3:  # Threshold for relevance
                    props = element.get('properties', {})
                    name = props.get('Name', 'Unknown')
                    etype = element.get('entity_type', 'Unknown')
                    
                    seen_ids.add(element_id)
                    results.append(self.format_result(
                        content=f"[IFC Element] {etype} '{name}'",
                        source="L1_ifc",
                        layer=1,
                        score=score,
                        metadata={
                            "id": element_id,
                            "name": name,
                            "type": etype,
                            "properties": props,
                            "match_method": "keyword"
                        }
                    ))
        
        # === STRATEGY 3: Semantic search via vector store ===
        if len(results) < top_k:
            semantic_results = self.vector_store.hybrid_search(query, top_k * 2, alpha=0.7)
            
            for res in semantic_results:
                metadata = res['metadata']
                doc_id = metadata.get('id')
                
                # Filter to L1 only and avoid duplicates
                if metadata.get('layer') == 'L1' and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    
                    # Boost score if this is a main element chunk
                    boost = 1.2 if metadata.get('type') == 'element_main' else 1.0
                    
                    results.append(self.format_result(
                        content=res['content'],
                        source="L1_ifc",
                        layer=1,
                        score=res['relevance_score'] * boost,
                        metadata=metadata
                    ))
        
        # === STRATEGY 4: Type-based retrieval (for "all walls" queries) ===
        if any(word in query_lower for word in ['all', 'list', 'show']):
            # Extract element type from query
            element_types = ['wall', 'beam', 'column', 'slab', 'door', 'window', 'roof']
            query_type = None
            for etype in element_types:
                if etype in query_lower:
                    query_type = etype
                    break
            
            if query_type:
                for element in self.l1_data:
                    element_id = element.get('id', 'unknown')
                    if element_id in seen_ids:
                        continue
                    
                    etype = element.get('entity_type', '').lower()
                    if query_type in etype:
                        props = element.get('properties', {})
                        name = props.get('Name', 'Unknown')
                        
                        seen_ids.add(element_id)
                        results.append(self.format_result(
                            content=f"[IFC Element] {element.get('entity_type')} '{name}'",
                            source="L1_ifc",
                            layer=1,
                            score=0.6,  # Base score for type matches
                            metadata={
                                "id": element_id,
                                "name": name,
                                "type": element.get('entity_type'),
                                "match_method": "type_filter"
                            }
                        ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"   Retrieved {len(results)} L1 results (top score: {results[0]['relevance_score']:.2f} if results else 0)")
        
        return results[:top_k]
    
    def get_element_by_name(self, name):
        """Direct lookup by element name"""
        for element in self.l1_data:
            props = element.get('properties', {})
            if props.get('Name') == name:
                return element
        return None
    
    def get_elements_by_type(self, element_type):
        """Get all elements of a given type"""
        results = []
        for element in self.l1_data:
            if element_type.lower() in element.get('entity_type', '').lower():
                results.append(element)
        return results