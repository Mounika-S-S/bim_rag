# src/retrieval/query_router.py
"""
Main query router - orchestrates all routers and retrievers
Handles L1, L2, L4, L5, L1L2L4, L1L2L5, L4L5 queries
"""
import re
import logging
from src.rag.semantic_router_v2 import SemanticRouterV2
from src.rag.semantic_router_l5 import SemanticRouterL5

# Import all retrievers
from src.retrieval.l1_retriever import L1Retriever
from src.retrieval.l2_retriever import L2Retriever
from src.retrieval.l4_retriever import L4Retriever
from src.retrieval.l5_retriever import L5Retriever
from src.retrieval.l1_l2_l4_retriever import L1L2L4Retriever
from src.retrieval.l1_l2_l5_retriever import L1L2L5Retriever
from src.retrieval.l4_l5_retriever import L4L5Retriever

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Master router that coordinates all routing and retrieval
    Supports:
    - L1 only (walls, beams, columns)
    - L2 only (products)
    - L4 only (regulations)
    - L5 only (requirements)
    - L1+L2+L4 (regulatory compliance)
    - L1+L2+L5 (project compliance)
    - L4+L5 (regulation vs requirement comparison)
    """
    
    # Keyword-based routing as fallback
    ROUTE_KEYWORDS = {
        "l1": ["wall", "walls", "beam", "column", "slab", "floor", "roof", 
               "ifc", "element", "building", "structure", "dimension", "height", "width"],
        "l2": ["product", "products", "material", "manufacturer", "brand",
               "siniat", "gyproc", "knauf", "rockwool", "paroc", "cost", "price", "catalog"],
        "l4": ["regulation", "regulations", "code", "nbc", "fire code", 
               "building code", "standard", "compliance", "requirement", "rule"],
        "l5": ["requirement", "requirements", "req", "spec", "specification",
               "project requirement", "rule", "mandatory", "priority", "client"],
          "l1_l2_l4": [
        "code violation", "regulatory", "fire code", "against code", 
        "nbc", "building code", "compliance with code",
        "violate height", "violate fire", "non-compliant with code",
        "walls that violate", "elements that fail", "non-compliant",
        "regulatory check", "code compliance", "regulation violation",
        "fails code", "does not meet code", "below code"
    ],
    "l1_l2_l5": [
        "non compliant", "non-compliant", "violation", "does not meet",
        "fails requirement", "against specification", "not compliant",
        "compliance check", "products do not meet", "fail the requirement",
        "requirement violation", "specification check", "project compliance",
        "check compliance", "verify requirements", "quality check"
    ]
      ,  "l4_l5": [
            "compare", "vs", "versus", "difference", "gap",
            "stricter", "weaker", "stronger", "exceed",
            "regulation vs requirement", "code vs rule",
            "project vs code", "requirement vs regulation",
            "does rule meet code", "is project stricter",
            "requirements vs regulations", "rules vs codes",
            "compliance gap", "missing requirement",
            # ADDED MORE KEYWORDS
            "above minimum code", "exceeds code", "exceed code",
            "more than code", "less than code", "below code",
            "meets code", "matches code", "equal to code",
            "code compliance", "regulation compliance",
            "comparison record", "relationship", "l4-l5"
        ]
    }
    
    # NEW: Relationship-specific keywords for boosting
    RELATIONSHIP_KEYWORDS = {
        "stricter": ["stricter", "exceeds", "above", "more than", "higher than", "stronger"],
        "weaker": ["weaker", "below", "less than", "lower than", "falls short"],
        "equal": ["equal", "matches", "meets", "same as", "compliant"],
        "missing": ["missing", "gap", "not addressed", "no requirement"]
    }
    
    def __init__(self, project_id):
        """
        Initialize router with project ID
        """
        self.project_id = project_id
        
        # Initialize semantic routers
        logger.info("🔄 Initializing semantic routers...")
        self.semantic_router = SemanticRouterV2()
        self.l5_router = SemanticRouterL5()
        
        # Initialize all retrievers
        logger.info("🔄 Initializing retrievers...")
        self.retrievers = {
            "l1": L1Retriever(project_id),
            "l2": L2Retriever(project_id),
            "l4": L4Retriever(project_id),
            "l5": L5Retriever(project_id),
            "l1_l2_l4": L1L2L4Retriever(project_id),
            "l1_l2_l5": L1L2L5Retriever(project_id),
            "l4_l5": L4L5Retriever(project_id)  # NEW
        }
        
        logger.info(f"✅ QueryRouter initialized for project: {project_id}")
        logger.info(f"   Available retrievers: {list(self.retrievers.keys())}")
    
    def route(self, query):
        """
        Determine which retriever to use
        
        Returns:
            tuple: (retriever_key, confidence, route_info)
        """
        query_lower = query.lower()
        logger.debug(f"Routing query: '{query[:50]}...'")
        
        # ============ STRATEGY 1: Check for L4-L5 comparison keywords ============
        # Give extra weight to L4-L5 queries
        comparison_keywords = [
            "compare", "vs", "versus", "difference", "gap",
            "stricter", "weaker", "regulation vs", "code vs",
            "requirements vs regulations", "rules vs codes",
            "above minimum code", "exceeds code", "exceed code",
            "meets code", "matches code", "equal to code",
            "comparison record", "relationship", "l4-l5"
        ]
        
        comparison_score = 0
        for keyword in comparison_keywords:
            if keyword in query_lower:
                comparison_score += 1
                logger.debug(f"   → L4-L5 keyword match: {keyword}")
        
        # If multiple comparison keywords, route to l4_l5 with high confidence
        if comparison_score >= 1:
            confidence = "high" if comparison_score >= 2 else "medium"
            logger.info(f"   → L4-L5 comparison match (score: {comparison_score})")
            return "l4_l5", confidence, {"method": "comparison_keyword", "score": comparison_score}
        
        # ============ STRATEGY 2: Check for compliance keywords ============
        compliance_keywords = {
            "l1_l2_l4": ["code violation", "regulatory", "fire code", "against code", 
                         "nbc", "building code", "compliance with code"],
            "l1_l2_l5": ["non compliant", "non-compliant", "violation", "does not meet",
                         "fails", "against requirement", "not compliant", "compliance check"]
        }
        
        for route, keywords in compliance_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    logger.info(f"   → Compliance keyword match: {route}")
                    return route, "keyword_high", {"method": "compliance_keyword", "matched": keyword}
        
        # ============ STRATEGY 3: Try L5 semantic router (specialized) ============
        try:
            l5_result = self.l5_router.route_query(query)
            if l5_result['confidence'] in ['high', 'medium']:
                # Map L5 router output to retriever keys
                route_map = {
                    "layer1_query": "l1",
                    "layer2_query": "l2",
                    "layer5_query": "l5",
                    "compliance_query": "l1_l2_l5",
                    "l1_l5_compliance": "l1_l2_l5",
                    "mixed_query": "l1"
                }
                if l5_result['primary_route'] in route_map:
                    logger.info(f"   → L5 router: {l5_result['primary_route']} ({l5_result['confidence']})")
                    return (route_map[l5_result['primary_route']], 
                            l5_result['confidence'], 
                            {"method": "l5_router", "details": l5_result})
        except Exception as e:
            logger.warning(f"L5 router failed: {e}")
        
        # ============ STRATEGY 4: Try main semantic router (V2) ============
        try:
            v2_result = self.semantic_router.route_query(query)
            if v2_result['confidence'] in ['high', 'medium']:
                route_map = {
                    "layer1_query": "l1",
                    "layer2_query": "l2",
                    "layer4_query": "l4",
                    "layer5_query": "l5",
                    "cross_layer_L1_L2_L4": "l1_l2_l4",
                    "cross_layer_L1_L2_L5": "l1_l2_l5",
                    "cross_layer_L4_L5": "l4_l5",  # Map to L4L5 retriever
                    "mixed_query": "l1"
                }
                if v2_result['primary_route'] in route_map:
                    logger.info(f"   → V2 router: {v2_result['primary_route']} ({v2_result['confidence']})")
                    return (route_map[v2_result['primary_route']], 
                            v2_result['confidence'], 
                            {"method": "v2_router", "details": v2_result})
        except Exception as e:
            logger.warning(f"V2 router failed: {e}")
        
        # ============ STRATEGY 5: Keyword-based fallback ============
        scores = {key: 0 for key in self.ROUTE_KEYWORDS}
        
        # Score each route based on keyword matches
        for route, keywords in self.ROUTE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[route] += 1
                    # Bonus for exact matches at start
                    if query_lower.startswith(keyword):
                        scores[route] += 2
                    # Extra bonus for l4_l5 route on relationship queries
                    if route == "l4_l5":
                        for rel_keywords in self.RELATIONSHIP_KEYWORDS.values():
                            if any(rel_kw in query_lower for rel_kw in rel_keywords):
                                scores[route] += 3
                                break
        
        # Find best route
        best_route = max(scores, key=scores.get)
        if scores[best_route] > 0:
            logger.info(f"   → Keyword match: {best_route} (score: {scores[best_route]})")
            return best_route, "keyword", {"method": "keyword", "scores": scores}
        
        # ============ STRATEGY 6: Default fallback ============
        logger.info(f"   → Default fallback: l1")
        return "l1", "very_low", {"method": "default"}
    
    def retrieve(self, query, top_k=5):
        """
        Main retrieval method - routes query and gets results
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            Dict with routing info and results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*60}")
        
        # Route the query
        retriever_key, confidence, route_info = self.route(query)
        
        logger.info(f"📌 ROUTED TO: {retriever_key} (confidence: {confidence})")
        
        # Get the appropriate retriever
        retriever = self.retrievers.get(retriever_key)
        if not retriever:
            logger.warning(f"Retriever {retriever_key} not found, falling back to l1")
            retriever = self.retrievers["l1"]
        
        # Retrieve results
        try:
            results = retriever.retrieve(query, top_k)
            
            # NEW: Post-process results for L4-L5 queries to boost comparison records
            if retriever_key == "l4_l5" and results:
                for result in results:
                    # Check if this is a comparison record (has relationship field)
                    metadata = result.get('metadata', {})
                    if metadata.get('layer') == 'comparison' and metadata.get('type') == 'l4_l5':
                        # Boost score for comparison records
                        result['relevance_score'] = min(1.0, result['relevance_score'] * 1.3)
                        
                        # Add relationship info to result for display
                        result['relationship'] = metadata.get('relationship', 'unknown')
            
            # Sort by boosted score
            results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"✅ Retrieved {len(results)} results")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            results = []
        
        return {
            "query": query,
            "route": {
                "retriever": retriever_key,
                "confidence": confidence,
                "details": route_info
            },
            "results": results,
            "total_found": len(results)
        }
    
    def get_retriever_stats(self):
        """Get statistics about retrievers"""
        stats = {}
        for name, retriever in self.retrievers.items():
            stats[name] = {
                "type": retriever.__class__.__name__,
                "has_data": hasattr(retriever, 'data') and len(getattr(retriever, 'data', [])) > 0
            }
        return stats
    
    def get_route_explanation(self, query):
        """Get detailed explanation of routing decision"""
        retriever_key, confidence, route_info = self.route(query)
        
        # Get semantic router explanation if available
        v2_explanation = ""
        if route_info.get('method') == 'v2_router':
            v2_explanation = self.semantic_router.get_route_explanation(query)
        
        explanation = f"\n{'='*60}\n"
        explanation += f"ROUTING DECISION\n"
        explanation += f"{'='*60}\n"
        explanation += f"Query: '{query}'\n"
        explanation += f"Retriever: {retriever_key}\n"
        explanation += f"Confidence: {confidence}\n"
        explanation += f"Method: {route_info.get('method', 'unknown')}\n"
        
        if v2_explanation:
            explanation += f"\nSemantic Router Details:\n{v2_explanation}\n"
        
        return explanation