from sentence_transformers import CrossEncoder
import numpy as np
from src.rag.unified_vector_store import UnifiedVectorStore
from src.embedding.vector_store import VectorStore
from src.core.model_manager import model_manager

class Retriever:
    def __init__(self):
        # Lazy-load cross-encoder only when reranking is needed
        self.reranker = None
        self.unified_store = None
        self.chroma_store = None

    def _get_reranker(self):
        """Lazy load the cross-encoder reranker"""
        if self.reranker is None:
            print("Loading reranker model...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✓ Reranker loaded!")
        return self.reranker

    def retrieve(self, query, routing_decision):
        """
        Retrieve relevant documents based on routing decision
        """
        strategy = routing_decision["strategy"]
        k = routing_decision["k"]
        use_reranking = routing_decision.get("use_reranking", False)

        if strategy == "unified_vector_store":
            return self._retrieve_unified(query, k, use_reranking)
        elif strategy == "chroma_vector_store":
            return self._retrieve_chroma(query, k)
        elif strategy == "deterministic_engine":
            return self._retrieve_deterministic(query)
        else:
            # Fallback
            return self._retrieve_unified(query, k, use_reranking)

    def _retrieve_unified(self, query, k, use_reranking):
        """Retrieve from FAISS unified store with optional reranking"""
        if not self.unified_store:
            raise ValueError("Unified vector store not loaded")

        # Get more candidates for reranking
        candidate_k = k * 4 if use_reranking else k
        candidates = self.unified_store.search(query, k=candidate_k)

        if not use_reranking or len(candidates) <= k:
            return candidates[:k]

        # Rerank using cross-encoder
        query_doc_pairs = [[query, doc] for doc in candidates]
        reranker = self._get_reranker()
        scores = reranker.predict(query_doc_pairs)

        # Sort by reranker scores (higher is better)
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:k]]

    def _retrieve_chroma(self, query, k):
        """Retrieve from ChromaDB for property lookups"""
        if not self.chroma_store:
            raise ValueError("Chroma vector store not loaded")

        results = self.chroma_store.query(query, n_results=k)
        return results.get("documents", [[]])[0] if results.get("documents") else []

    def _retrieve_deterministic(self, query):
        """For compliance checks, return minimal context"""
        # This could integrate with the compliance engine
        return ["Use deterministic compliance checking for precise rule validation."]

    def load_unified_store(self, path):
        """Load the unified FAISS vector store"""
        self.unified_store = UnifiedVectorStore()
        self.unified_store.load(path)

    def load_chroma_store(self, project_id):
        """Optional: skip ChromaDB when not using it."""
        # Chroma is not used in FAISS-only mode. Keep this as no-op.
        self.chroma_store = None

    def clear_chroma_store(self):
        """Explicitly drop Chroma store if exists."""
        self.chroma_store = None