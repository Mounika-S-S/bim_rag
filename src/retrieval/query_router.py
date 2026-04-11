# src/retrieval/query_router.py
# Semantic 6-target router: L1 | L2 | L3 | L4 | L5 | Compliance

from src.core.model_manager import model_manager
import numpy as np


ROUTE_INTENTS = {
    "L1": [
        "IFC element properties",
        "what is the beam dimension",
        "show me the slab element",
        "list all IFC objects",
        "element type in model",
        "BIM model component",
        "which elements are in the building",
        "IfcBeam IfcSlab IfcColumn",
        "IFC global id",
        "building model element",
    ],
    "L2": [
        "product specifications",
        "material grade",
        "fire rating of the product",
        "product name",
        "reinforced concrete beam product",
        "what product is used",
        "product data sheet",
        "unit cost of material",
        "product category",
        "L2 product library",
    ],
    "L3": [
        "construction process",
        "installation method",
        "process rule",
        "how should this be constructed",
        "quality control procedure",
        "work method statement",
        "execution sequence",
        "process specification",
        "construction method",
        "on-site process",
    ],
    "L4": [
        "regulation clause",
        "national building code",
        "TANGEDCO standard",
        "fire and rescue requirement",
        "Tamil Nadu building rules",
        "BIS standard IS 800",
        "setback requirement",
        "road width regulation",
        "basement height rule",
        "legal requirement from code book",
    ],
    "L5": [
        "company requirement",
        "project specific rule",
        "client requirement",
        "BOQ item",
        "rate schedule",
        "item code description",
        "project specification",
        "employer requirement",
        "L5 requirement",
        "unit rate",
    ],
    "Compliance": [
        "is this compliant",
        "compliance check",
        "which elements are non-compliant",
        "why is it non-compliant",
        "what value is required for compliance",
        "violation found",
        "does this meet the regulation",
        "compliance status",
        "explain the compliance issue",
        "fire rating compliance",
        "non-compliant elements",
        "regulation vs actual value",
        "compliance report",
        "why was this flagged",
        "which layer caused non-compliance",
    ],
}

# Maps each route to which vectorDB tag prefix to bias toward
ROUTE_TAG_PREFIX = {
    "L1":         "[IFC Element]",
    "L2":         "[Product]",
    "L3":         "[Process Rule]",
    "L4":         "[Regulation]",
    "L5":         "[Requirement]",
    "Compliance": "[Compliance:",
}


class QueryRouter:
    def __init__(self):
        self.model = model_manager.get_model("all-mpnet-base-v2")
        self._build_centroids()

    def _build_centroids(self):
        self.centroids = {}
        for route, examples in ROUTE_INTENTS.items():
            embeddings = self.model.encode(examples)
            centroid = np.mean(embeddings, axis=0)
            # L2-normalize so cosine sim = dot product
            norm = np.linalg.norm(centroid)
            self.centroids[route] = centroid / norm if norm > 0 else centroid

    def classify_query(self, query: str):
        """
        Returns (route: str, confidence: float, all_scores: dict)
        Route is one of: L1, L2, L3, L4, L5, Compliance
        """
        q_emb = self.model.encode([query])[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = {}
        for route, centroid in self.centroids.items():
            scores[route] = float(np.dot(q_emb, centroid))

        best_route = max(scores, key=scores.get)
        best_score = scores[best_route]
        return best_route, best_score, scores

    def route_query(self, query: str, mode: str = "faiss") -> dict:
        """
        Returns routing decision dict used by Retriever.
        """
        route, confidence, all_scores = self.classify_query(query)

        routing = {
            "route": route,                          # NEW: L1/L2/L3/L4/L5/Compliance
            "confidence": confidence,
            "all_scores": all_scores,
            "strategy": "unified_vector_store",
            "tag_prefix": ROUTE_TAG_PREFIX.get(route, ""),
            "k": 5,
            "use_reranking": False,
        }

        # Tune retrieval depth per route
        if route == "Compliance":
            routing["k"] = 8
            routing["use_reranking"] = True   # compliance needs precision
        elif route in ("L4", "L5"):
            routing["k"] = 6
        elif route == "L1":
            routing["k"] = 5

        return routing
