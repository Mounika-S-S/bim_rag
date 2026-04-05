# src/retrieval/semantic_router.py
"""
Router for semantic queries - routes to FAISS vector store (L1-L5 only)
"""

import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


class SemanticRouter:
    """
    Router for semantic queries.
    Routes queries to FAISS vector store containing only L1-L5 data.
    """
    
    def __init__(self, project_id: str, embedding_model_name: str = "all-mpnet-base-v2"):
        """
        Args:
            project_id: Project identifier
            embedding_model_name: MUST match the model used to build the FAISS index!
                                 Default: "all-mpnet-base-v2" (768-dim)
                                 Alternative: "all-MiniLM-L6-v2" (384-dim)
        """
        self.project_id = project_id
        self.base_path = f"data/processed/{project_id}"
        
        # Load embedding model - MUST match index dimension
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Load FAISS index
        self.index_path = f"{self.base_path}/unified.index"
        self.texts_path = f"{self.base_path}/unified.index.texts"
        
        self.index = None
        self.text_chunks = []
        
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self._load_index()
        else:
            print(f"⚠️ Vector store not found at {self.index_path}")
            print("   Build vector store first using option 11")
    
    def _load_index(self):
        """Load FAISS index and text chunks"""
        self.index = faiss.read_index(self.index_path)
        with open(self.texts_path, 'r', encoding='utf-8') as f:
            self.text_chunks = json.load(f)
        
        # Verify dimension
        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"✓ FAISS index dimension: {self.index.d}")
        print(f"✓ Embedding model dimension: {embedding_dim}")
        
        if self.index.d != embedding_dim:
            raise ValueError(
                f"Dimension mismatch! FAISS index: {self.index.d}, "
                f"Embedding model: {embedding_dim}. "
                f"Please rebuild vector store or use matching embedding model."
            )
        
        print(f"✓ Loaded FAISS index with {len(self.text_chunks)} chunks")
    
    def route_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Route query to FAISS vector store (L1-L5 only)
        """
        if self.index is None:
            return {
                "source": "error",
                "results": [],
                "retrieved_ids": [],
                "retrieved_docs": [],
                "confidence": 0.0,
                "error": "Vector store not loaded"
            }
        
        # Encode query
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get results
        results = []
        retrieved_ids = []
        retrieved_docs = []
        
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                chunk = self.text_chunks[idx]
                results.append(chunk)
                retrieved_docs.append(chunk)
                retrieved_ids.append(f"chunk_{idx}")
        
        return {
            "source": "semantic_vector_store",
            "results": results,
            "retrieved_ids": retrieved_ids,
            "retrieved_docs": retrieved_docs,
            "confidence": 0.85,
            "distances": distances[0].tolist()
        }
    
    def get_all_docs(self) -> List[str]:
        """Get all document texts for ground truth"""
        return self.text_chunks
    
    def get_doc_ids(self) -> List[str]:
        """Get all document IDs for ground truth"""
        return [f"chunk_{i}" for i in range(len(self.text_chunks))]