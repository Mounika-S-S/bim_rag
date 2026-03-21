from sentence_transformers import SentenceTransformer
import numpy as np
from src.core.model_manager import model_manager

class QueryRouter:
    def __init__(self):
        # Use shared model manager to avoid repeated downloads
        self.model = model_manager.get_model("all-mpnet-base-v2")

        # Define intent categories with example queries
        self.intents = {
            "compliance_check": [
                "is this compliant",
                "check compliance",
                "violation check",
                "does this meet requirements",
                "compliance status"
            ],
            "property_lookup": [
                "what is the fire rating",
                "beam dimensions",
                "product specifications",
                "material properties",
                "technical details"
            ],
            "inference_explanation": [
                "why is this non-compliant",
                "explain the issue",
                "reason for violation",
                "inference details",
                "how was this determined"
            ],
            "general_query": [
                "tell me about",
                "what are",
                "describe",
                "explain"
            ]
        }

        # Pre-compute intent embeddings
        self.intent_embeddings = {}
        for intent, examples in self.intents.items():
            embeddings = self.model.encode(examples)
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)

    def classify_query(self, query):
        """
        Classify query intent using semantic similarity
        Returns: (intent, confidence_score)
        """
        query_embedding = self.model.encode([query])[0]

        best_intent = None
        best_score = -1

        for intent, intent_emb in self.intent_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, intent_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(intent_emb)
            )

            if similarity > best_score:
                best_score = similarity
                best_intent = intent

        return best_intent, best_score

    def route_query(self, query, mode="faiss"):
        """
        Route query to appropriate retrieval strategy.
        Mode options: "faiss" (default), "hybrid".
        """
        intent, confidence = self.classify_query(query)

        routing = {
            "intent": intent,
            "confidence": confidence,
            "strategy": "unified_vector_store",  # default to FAISS
            "k": 5,
            "use_reranking": False
        }

        if intent == "compliance_check":
            routing["strategy"] = "deterministic_engine"
            routing["k"] = 3
        elif intent == "property_lookup" and mode == "hybrid":
            routing["strategy"] = "chroma_vector_store"
            routing["k"] = 10
        elif intent == "inference_explanation":
            routing["strategy"] = "unified_vector_store"
            routing["k"] = 8
            routing["use_reranking"] = True

        # Force FAISS-only mode if requested
        if mode == "faiss" and routing["strategy"] == "chroma_vector_store":
            routing["strategy"] = "unified_vector_store"

        return routing