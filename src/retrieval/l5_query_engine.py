# src/retrieval/l5_query_engine.py
"""
Complete query engine for L1, L2, L5 individual and compliance queries
Integrates semantic router + hybrid retrieval
Based on multi-agent RAG research [citation:4][citation:6]
"""
from src.rag.semantic_router_l5 import SemanticRouterL5
from src.rag.l5_vector_store import L5VectorStore
from src.core.json_storage import JSONStorage
import logging

logger = logging.getLogger(__name__)


class L5QueryEngine:
    """
    Complete query engine for L1-L2-L5 queries
    Uses semantic routing + hybrid retrieval
    """
    
    def __init__(self, project_id):
        self.project_id = project_id
        self.router = SemanticRouterL5()
        self.vector_store = L5VectorStore(project_id)
        self.vector_store.load(f"data/processed/{project_id}/l5_hybrid.index")
        
        # Load data for direct lookup
        self.l1 = JSONStorage.load(project_id, "L1_ifc.json") or []
        self.l2 = JSONStorage.load(project_id, "L2_product.json") or []
        self.l5 = JSONStorage.load(project_id, "L5_requirement.json") or []
        self.compliance = JSONStorage.load(project_id, "compliance_l5.json") or []
        
        logger.info(f"✅ L5QueryEngine initialized for project {project_id}")
    
    def query(self, user_query, top_k=5):
        """
        Main query method - routes and retrieves
        """
        # Step 1: Route the query
        route_info = self.router.route_query(user_query)
        primary_route = route_info["primary_route"]
        
        print(f"\n🔍 Query: '{user_query}'")
        print(f"   Routed to: {primary_route} (confidence: {route_info['confidence']}, score: {route_info['score']:.3f})")
        
        # Step 2: Route to appropriate handler
        if primary_route == "layer1_query":
            results = self._handle_l1_query(user_query, top_k)
        elif primary_route == "layer2_query":
            results = self._handle_l2_query(user_query, top_k)
        elif primary_route == "layer5_query":
            results = self._handle_l5_query(user_query, top_k)
        elif primary_route in ["compliance_query", "l1_l5_compliance"]:
            results = self._handle_compliance_query(user_query, top_k)
        else:
            results = self._handle_mixed_query(user_query, top_k)
        
        return {
            "query": user_query,
            "route_info": route_info,
            "results": results,
            "total_found": len(results)
        }
    
    def _handle_l1_query(self, query, top_k):
        """Handle L1-only queries with hybrid search"""
        # Try keyword matching first for exact IDs/names
        query_lower = query.lower()
        exact_matches = []
        
        for element in self.l1:
            props = element.get('properties', {})
            name = props.get('Name', '').lower()
            if name and name in query_lower:
                exact_matches.append({
                    "content": f"[IFC] {element.get('entity_type')} '{props.get('Name')}'",
                    "source": "L1_ifc",
                    "layer": 1,
                    "relevance_score": 1.0,
                    "metadata": element
                })
        
        if exact_matches:
            return exact_matches[:top_k]
        
        # Fallback to hybrid search
        return self.vector_store.hybrid_search(query, top_k, alpha=0.7)
    
    def _handle_l2_query(self, query, top_k):
        """Handle L2-only queries"""
        query_lower = query.lower()
        exact_matches = []
        
        for product in self.l2:
            props = product.get('properties', {})
            name = props.get('Product_Name', '').lower()
            mfr = props.get('Manufacturer', '').lower()
            
            if (name and name in query_lower) or (mfr and mfr in query_lower):
                exact_matches.append({
                    "content": f"[PRODUCT] {props.get('Product_Name')} from {props.get('Manufacturer')}",
                    "source": "L2_product",
                    "layer": 2,
                    "relevance_score": 0.9,
                    "metadata": product
                })
        
        if exact_matches:
            return exact_matches[:top_k]
        
        # Boost keyword weight for product queries
        return self.vector_store.hybrid_search(query, top_k, alpha=0.4)
    
    def _handle_l5_query(self, query, top_k):
        """Handle L5-only queries"""
        query_lower = query.lower()
        exact_matches = []
        
        for req in self.l5:
            props = req.get('properties', {})
            req_id = props.get('RequirementID', '').lower()
            
            if req_id and req_id in query_lower:
                rule = f"{props.get('Property', '')} {props.get('Operator', '')} {props.get('RequiredValue', '')} {props.get('Unit', '')}"
                exact_matches.append({
                    "content": f"[REQUIREMENT] {props.get('RequirementID')}: {props.get('Description', '')}",
                    "source": "L5_requirement",
                    "layer": 5,
                    "relevance_score": 1.0,
                    "metadata": {
                        "requirement_id": props.get('RequirementID'),
                        "rule": rule,
                        "priority": props.get('Priority', 'Medium')
                    }
                })
        
        if exact_matches:
            return exact_matches[:top_k]
        
        # Semantic search for requirement descriptions
        return self.vector_store.hybrid_search(query, top_k, alpha=0.8)
    
    def _handle_compliance_query(self, query, top_k):
        """
        Handle compliance queries using pre-computed results + hybrid search
        Based on multi-hop reasoning research [citation:6]
        """
        query_lower = query.lower()
        
        # First, check pre-computed compliance results
        compliance_results = []
        
        for issue in self.compliance:
            element_name = issue.get('element_name', '').lower()
            req_id = issue.get('requirement_id', '').lower()
            
            # If query mentions specific element or requirement
            if (element_name and element_name in query_lower) or (req_id and req_id in query_lower):
                status = "COMPLIANT" if issue.get('is_compliant') else "NON-COMPLIANT"
                compliance_results.append({
                    "content": f"[COMPLIANCE] {issue.get('element_name')} is {status}: {issue.get('comparison', '')}",
                    "source": "compliance_l5",
                    "layer": "compliance",
                    "relevance_score": 1.0,
                    "metadata": issue
                })
        
        if compliance_results:
            return compliance_results[:top_k]
        
        # For general compliance queries, use hybrid search with compliance boost
        results = self.vector_store.hybrid_search(query, top_k, alpha=0.6)
        
        # Filter to prioritize compliance chunks
        compliance_chunks = [r for r in results if r['metadata'].get('layer') == 'compliance']
        other_chunks = [r for r in results if r['metadata'].get('layer') != 'compliance']
        
        return (compliance_chunks + other_chunks)[:top_k]
    
    def _handle_mixed_query(self, query, top_k):
        """Handle mixed/general queries with balanced hybrid search"""
        return self.vector_store.hybrid_search(query, top_k, alpha=0.5)