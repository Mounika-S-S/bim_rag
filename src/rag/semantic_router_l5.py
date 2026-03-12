# src/rag/semantic_router_l5.py
"""
Specialized semantic router for L5 (Requirements) and compliance queries
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticRouterL5:
    """
    Specialized router for L5 requirements and compliance queries
    Focuses on project requirements and compliance checking
    """
    
    def __init__(self, config=None):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define route templates focused on L5
        self.route_templates = {
            # L1 Only queries
            "layer1_query": {
                "examples": [
                    "Show me all walls",
                    "List external walls",
                    "What beams are in the model?"
                ],
                "sources": ["L1_ifc"],
                "description": "IFC element queries"
            },
            
            # L2 Only queries
            "layer2_query": {
                "examples": [
                    "Show me products with EI120",
                    "List Siniat products",
                    "What is the cost?"
                ],
                "sources": ["L2_product"],
                "description": "Product queries"
            },
            
            # L5 Only queries
            "layer5_query": {
                "examples": [
                    "What are the project requirements?",
                    "Show me all rules",
                    "List fire rating requirements",
                    "What is REQ-FIRE-001?",
                    "Show high priority requirements",
                    "What are the mandatory rules?",
                    "Find requirements for external walls",
                    "List all specifications"
                ],
                "sources": ["L5_requirement"],
                "description": "Project requirement queries"
            },
            
            # L1-L2-L5 Compliance queries
            "compliance_query": {
                "examples": [
                    "Which walls don't meet requirements?",
                    "Show non-compliant elements",
                    "What fails the fire rating requirement?",
                    "Check compliance of external walls",
                    "Find violations of REQ-FIRE-001",
                    "List all compliance issues",
                    "Which elements are non-compliant?",
                    "Show me what needs fixing"
                ],
                "sources": ["mismatch_l5", "L1_ifc", "L5_requirement"],
                "description": "Compliance checking queries"
            },
            
            # L1-L5 specific compliance
            "l1_l5_compliance": {
                "examples": [
                    "Check walls against requirements",
                    "Do beams meet specifications?",
                    "Verify column compliance",
                    "Are external walls compliant?",
                    "Check fire rating on all walls"
                ],
                "sources": ["mismatch_l5", "L1_ifc", "L5_requirement"],
                "description": "Element-specific compliance checks"
            },
            
            # Mixed queries (fallback)
            "mixed_query": {
                "examples": [
                    "Tell me about fire safety",
                    "What are the specifications?",
                    "Show me building requirements",
                    "Information about walls and products"
                ],
                "sources": ["L1_ifc", "L2_product", "L5_requirement"],
                "description": "Mixed/general queries"
            }
        }
        
        # Pre-compute route embeddings
        self.route_embeddings = self._compute_route_embeddings()
        
        # Confidence thresholds
        self.high_confidence = 0.75
        self.medium_confidence = 0.6
        self.low_confidence = 0.4
    
    def _compute_route_embeddings(self):
        """Compute embeddings for each route"""
        route_embeddings = {}
        for route, config in self.route_templates.items():
            example_embeddings = self.encoder.encode(config["examples"])
            route_embeddings[route] = np.mean(example_embeddings, axis=0)
        return route_embeddings
    
    def route_query(self, query):
        """
        Route query to appropriate handler
        Special handling for compliance keywords
        """
        query_embedding = self.encoder.encode([query])[0]
        query_lower = query.lower()
        
        # Calculate similarities
        similarities = {}
        for route, route_emb in self.route_embeddings.items():
            similarity = cosine_similarity([query_embedding], [route_emb])[0][0]
            similarities[route] = float(similarity)
        
        # Boost compliance queries if keywords present
        compliance_keywords = ["non compliant", "violation", "fail", "check", "verify", "compliance"]
        if any(k in query_lower for k in compliance_keywords):
            if "compliance_query" in similarities:
                similarities["compliance_query"] *= 1.2
            if "l1_l5_compliance" in similarities:
                similarities["l1_l5_compliance"] *= 1.1
        
        # Find best route
        best_route = max(similarities, key=similarities.get)
        best_score = similarities[best_route]
        
        # Determine confidence
        if best_score >= self.high_confidence:
            confidence = "high"
        elif best_score >= self.medium_confidence:
            confidence = "medium"
        elif best_score >= self.low_confidence:
            confidence = "low"
        else:
            confidence = "very_low"
        
        return {
            "primary_route": best_route,
            "confidence": confidence,
            "score": best_score,
            "all_scores": similarities,
            "sources": self.route_templates[best_route]["sources"],
            "description": self.route_templates[best_route]["description"]
        }