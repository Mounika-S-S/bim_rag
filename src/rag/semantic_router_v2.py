# src/rag/semantic_router_v2.py
"""
ML-based semantic router with confidence scoring
Handles L1, L2, L4, L5, and cross-layer queries
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticRouterV2:
    """
    Production-grade semantic router for all layer types
    Uses sentence embeddings to understand query intent
    """
    
    def __init__(self, config=None):
        # Load embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define route templates with rich examples
        self.route_templates = {
            # ============ L1: IFC Elements ============
            "layer1_query": {
                "examples": [
                    "Show me all walls",
                    "List external walls",
                    "What beams are in the model?",
                    "Show columns with height > 3m",
                    "Find load-bearing walls",
                    "List all IFC elements",
                    "What are the wall dimensions?",
                    "Show me structural elements",
                    "Display all slabs",
                    "Count the number of doors"
                ],
                "sources": ["L1_ifc"],
                "description": "Queries about IFC building elements"
            },
            
            # ============ L2: Products ============
            "layer2_query": {
                "examples": [
                    "Show me products with EI120",
                    "List all Siniat products",
                    "What is the cost of fire board?",
                    "Find products by manufacturer",
                    "Show product specifications",
                    "List fire-rated systems",
                    "What products are available?",
                    "Compare Siniat and Gyproc",
                    "Tell me about acoustic panels",
                    "What insulation materials exist?"
                ],
                "sources": ["L2_product"],
                "description": "Queries about construction products"
            },
            
            # ============ L4: Regulations ============
            "layer4_query": {
                "examples": [
                    "What does NBC say about fire?",
                    "Show fire code requirements",
                    "What is the regulation for EI120?",
                    "Building code fire ratings",
                    "What are the minimum setbacks?",
                    "Tell me about fire safety rules",
                    "What does the code say about stairs?",
                    "Regulations for external walls",
                    "Fire resistance requirements",
                    "Building height limitations"
                ],
                "sources": ["L4_regulation"],
                "description": "Queries about building regulations and codes"
            },
            
            # ============ L5: Project Requirements ============
            "layer5_query": {
                "examples": [
                    "What are the project requirements?",
                    "Show me all rules",
                    "List fire rating requirements",
                    "What is REQ-FIRE-001?",
                    "Show high priority requirements",
                    "What are the mandatory rules?",
                    "Find requirements for external walls",
                    "List all specifications",
                    "Tell me about quality standards",
                    "What does the client require?"
                ],
                "sources": ["L5_requirement"],
                "description": "Queries about project requirements and specifications"
            },
            
            # ============ L1+L2+L4: Regulatory Compliance ============
            "cross_layer_L1_L2_L4": {
                "examples": [
                    "Which walls violate fire code?",
                    "Show code non-compliance",
                    "Walls not meeting regulations",
                    "Find regulatory violations",
                    "Check against NBC requirements",
                    "What elements fail building code?",
                    "Show me fire code violations",
                    "Which products don't meet standards?",
                    "Regulatory compliance issues",
                    "What needs to be fixed for code?"
                ],
                "sources": ["mismatch", "L1_ifc", "L2_product", "L4_regulation"],
                "description": "Regulatory compliance checking (L1+L2+L4)"
            },
            
            # ============ L1+L2+L5: Project Compliance ============
            "cross_layer_L1_L2_L5": {
                "examples": [
                    "Which walls don't meet requirements?",
                    "Show non-compliant external walls",
                    "What walls fail fire rating?",
                    "Find mismatches between model and rules",
                    "Walls that violate project specs",
                    "List all compliance issues",
                    "Show me what needs fixing",
                    "Which elements are non-compliant?",
                    "Project requirement violations",
                    "Check compliance with specifications"
                ],
                "sources": ["mismatch_l5", "L1_ifc", "L2_product", "L5_requirement"],
                "description": "Project compliance checking (L1+L2+L5)"
            },
            
            # ============ L4+L5: Requirement vs Regulation ============
            "cross_layer_L4_L5": {
                "examples": [
                    "Compare rules with regulations",
                    "Is project stricter than code?",
                    "Show differences between L4 and L5",
                    "Requirement vs regulation comparison",
                    "Does rule meet code?",
                    "Compare project specs with building code",
                    "What are the gaps between rules and code?",
                    "Where is project exceeding code?",
                    "Are requirements stricter than regulations?",
                    "Compare fire ratings: project vs code"
                ],
                "sources": ["mismatch", "L4_regulation", "L5_requirement"],
                "description": "Requirement vs regulation comparison"
            },
            
            # ============ Mixed/General Queries ============
            "mixed_query": {
                "examples": [
                    "Tell me about fire safety",
                    "What are the specifications?",
                    "Show me building requirements",
                    "Information about walls and products",
                    "Tell me everything about this project",
                    "What do I need to know?",
                    "Give me an overview",
                    "Show me all data",
                    "What's in the model?",
                    "Project summary"
                ],
                "sources": ["L1_ifc", "L2_product", "L4_regulation", "L5_requirement"],
                "description": "Mixed/general queries across multiple layers"
            }
        }
        
        # Pre-compute route embeddings
        self.route_embeddings = self._compute_route_embeddings()
        
        # Confidence thresholds
        self.high_confidence = 0.75
        self.medium_confidence = 0.6
        self.low_confidence = 0.4
        
        logger.info(f"✅ SemanticRouterV2 initialized with {len(self.route_templates)} routes")
    
    def _compute_route_embeddings(self):
        """Compute embeddings for each route by averaging example embeddings"""
        route_embeddings = {}
        
        for route, config in self.route_templates.items():
            # Encode all examples
            example_embeddings = self.encoder.encode(config["examples"])
            # Use mean embedding as route representation
            route_embeddings[route] = np.mean(example_embeddings, axis=0)
            
        return route_embeddings
    
    def route_query(self, query):
        """
        Route query to appropriate layer with confidence scores
        
        Args:
            query: Natural language query string
            
        Returns:
            Dict with routing information
        """
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Calculate similarity to each route
        similarities = {}
        for route, route_emb in self.route_embeddings.items():
            similarity = cosine_similarity([query_embedding], [route_emb])[0][0]
            similarities[route] = float(similarity)
        
        # Sort by similarity
        sorted_routes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get primary route and confidence
        primary_route, primary_score = sorted_routes[0]
        
        # Determine confidence level
        if primary_score >= self.high_confidence:
            confidence = "high"
        elif primary_score >= self.medium_confidence:
            confidence = "medium"
        elif primary_score >= self.low_confidence:
            confidence = "low"
        else:
            confidence = "very_low"
        
        # Get sources from primary route
        primary_sources = self.route_templates[primary_route]["sources"]
        
        # Get secondary routes (above medium confidence)
        secondary_routes = [
            route for route, score in sorted_routes[1:4]
            if score >= self.medium_confidence
        ]
        
        # If confidence is low, add secondary sources
        all_sources = primary_sources.copy()
        if confidence in ["low", "very_low"]:
            for route in secondary_routes:
                all_sources.extend(self.route_templates[route]["sources"])
        
        # Remove duplicates
        all_sources = list(set(all_sources))
        
        return {
            "primary_route": primary_route,
            "confidence": confidence,
            "score": primary_score,
            "all_scores": similarities,
            "sources": all_sources,
            "secondary_routes": secondary_routes,
            "description": self.route_templates[primary_route]["description"]
        }
    
    def get_route_explanation(self, query):
        """Get human-readable explanation of routing decision"""
        result = self.route_query(query)
        
        explanation = f"Query routed to: {result['primary_route']}\n"
        explanation += f"Confidence: {result['confidence']} (score: {result['score']:.3f})\n"
        explanation += f"Description: {result['description']}\n"
        explanation += f"Sources: {', '.join(result['sources'])}\n"
        
        if result['secondary_routes']:
            explanation += f"Secondary matches: {', '.join(result['secondary_routes'])}"
        
        return explanation