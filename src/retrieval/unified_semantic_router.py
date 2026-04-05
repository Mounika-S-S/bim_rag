# src/retrieval/semantic_router.py
"""
Semantic Router - Routes queries to appropriate data source
- Compliance queries → JSON files (l124/l125/l45)
- Other queries → Vector DB (L1, L2, L3, L4, L5 ONLY)
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss


class UnifiedSemanticRouter:
    """
    Router that handles:
    - Stage 1: JSON only (compliance queries)
    - Stage 2: Intent-based (compliance → JSON, others → Vector DB)
    - Stage 3: Vector DB only (all queries)
    """
    
    def __init__(self, project_id: str, embedding_model_name: str = "all-mpnet-base-v2"):
        """
        Args:
            project_id: Project identifier
            embedding_model_name: Model for embeddings (must match FAISS index)
        """
        self.project_id = project_id
        self.base_path = f"data/processed/{project_id}"
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # =========================================================
        # 1. DETERMINISTIC DATA (Compliance JSON files)
        # =========================================================
        
        self.l124_data = self._load_json("l124_inference.json")
        self.l125_data = self._load_json("l125_inference.json")
        self.l45_data = self._load_json("l45_inference.json")
        
        self.all_compliance_data = self.l124_data + self.l125_data + self.l45_data
        
        print(f"   Loaded compliance data: {len(self.l124_data)} l124, {len(self.l125_data)} l125, {len(self.l45_data)} l45")
        
        # =========================================================
        # 2. SEMANTIC DATA (L1-L5 ONLY - No inference chunks)
        # =========================================================
        # IMPORTANT: This vector store should ONLY contain L1-L5 data
        # We build a separate index for L1-L5 only (not unified.index)
        
        self.l1l5_index_path = f"{self.base_path}/l1l5_only.index"
        self.l1l5_texts_path = f"{self.base_path}/l1l5_only.index.texts"
        
        self.index = None
        self.text_chunks = []
        
        # Check if L1-L5 only index exists
        if os.path.exists(self.l1l5_index_path) and os.path.exists(self.l1l5_texts_path):
            self._load_l1l5_index()
        else:
            print(f"⚠️ L1-L5 vector store not found at {self.l1l5_index_path}")
            print("   Building L1-L5 only index from layer files...")
            self._build_l1l5_index()
    
    def _load_json(self, filename: str) -> List[Dict]:
        """Load JSON file if exists"""
        filepath = os.path.join(self.base_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _build_l1l5_index(self):
        """
        Build FAISS index with ONLY L1-L5 data (no inference chunks)
        Reads directly from L1_ifc.json, L2_product.json, L3_process.json,
        L4_regulation.json, L5_requirement.json
        """
        print("   Building L1-L5 only vector store...")
        
        knowledge_chunks = []
        
        # Load L1 IFC Elements
        l1 = self._load_json("L1_ifc.json")
        for element in l1:
            props = element.get("properties", {})
            name = props.get("Name", "Unknown Element")
            text = f"[IFC Element] {name}. "
            for k, v in props.items():
                text += f"{k} = {v}. "
            knowledge_chunks.append(text.strip())
        
        # Load L2 Products
        l2 = self._load_json("L2_product.json")
        for product in l2:
            props = product.get("properties", {})
            name = props.get("Product_Name", props.get("text", "Unknown Product"))
            text = f"[Product] {name}. "
            for k, v in props.items():
                if k not in ["source_document", "row_number", "numeric_value"]:
                    text += f"{k} = {v}. "
            knowledge_chunks.append(text.strip())
        
        # Load L3 Process Rules
        l3 = self._load_json("L3_process.json")
        for process in l3:
            text = process.get("properties", {}).get("text", "")
            if text and len(text) > 20:
                knowledge_chunks.append(f"[Process Rule] {text[:500]}")
        
        # Load L4 Regulations
        l4 = self._load_json("L4_regulation.json")
        for rule in l4:
            text = rule.get("properties", {}).get("text", rule.get("text", ""))
            if text and len(text) > 20:
                knowledge_chunks.append(f"[Regulation] {text[:500]}")
        
        # Load L5 Requirements
        l5 = self._load_json("L5_requirement.json")
        for req in l5:
            props = req.get("properties", {})
            code = props.get("item_code", "")
            desc = props.get("description", props.get("text", ""))
            if desc:
                text = f"[Requirement] {desc}. Code: {code}."
                if props.get("unit"):
                    text += f" Unit: {props.get('unit')}."
                if props.get("rate"):
                    text += f" Rate: {props.get('rate')}."
                knowledge_chunks.append(text)
        
        # Filter empty chunks
        knowledge_chunks = [c for c in knowledge_chunks if c and len(c.strip()) > 20]
        
        print(f"   Created {len(knowledge_chunks)} chunks from L1-L5 data")
        
        if not knowledge_chunks:
            print("   No L1-L5 chunks created!")
            self.index = None
            self.text_chunks = []
            return
        
        self.text_chunks = knowledge_chunks
        
        # Generate embeddings
        embeddings = self.embedder.encode(knowledge_chunks)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        
        # Build FAISS index
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128
        self.index.add(embeddings)
        
        # Save index
        faiss.write_index(self.index, self.l1l5_index_path)
        with open(self.l1l5_texts_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f)
        
        print(f"✓ Built L1-L5 FAISS index with {len(knowledge_chunks)} chunks")
    
    def _load_l1l5_index(self):
        """Load L1-L5 only FAISS index"""
        self.index = faiss.read_index(self.l1l5_index_path)
        with open(self.l1l5_texts_path, 'r', encoding='utf-8') as f:
            self.text_chunks = json.load(f)
        
        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        print(f"✓ L1-L5 FAISS index: {self.index.d} dim, {len(self.text_chunks)} chunks")
        
        if self.index.d != embedding_dim:
            raise ValueError(f"Dimension mismatch! FAISS index: {self.index.d}, Model: {embedding_dim}")
    
    def _is_compliance_query(self, query: str) -> bool:
        """Check if query is about compliance"""
        compliance_keywords = [
            'non-compliant', 'non compliant', 'violation', 'compliance issue',
            'fire rating violation', 'missing product', 'does not meet',
            'compliance status', 'check compliance', 'regulation violated'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in compliance_keywords)
    
    def _retrieve_from_compliance(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve from deterministic JSON files (l124/l125/l45)"""
        query_lower = query.lower()
        
        # Determine which compliance file to use
        if any(word in query_lower for word in ["beam", "yc-st-sf-bip", "structural"]):
            data = self.l124_data
            source = "l124"
        elif any(word in query_lower for word in ["requirement", "missing", "l125"]):
            data = self.l125_data
            source = "l125"
        elif any(word in query_lower for word in ["gap", "compare", "difference", "l45"]):
            data = self.l45_data
            source = "l45"
        else:
            data = self.all_compliance_data
            source = "all_compliance"
        
        # Keyword matching
        query_terms = set(query_lower.split())
        scored_results = []
        
        for item in data:
            text = str(item).lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored_results.append((score, item, source))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        retrieved_ids = []
        retrieved_docs = []
        
        for score, item, src in scored_results[:top_k]:
            doc_id = item.get("element_id", item.get("id", f"doc_{len(retrieved_ids)}"))
            doc_text = self._format_doc_text(item)
            retrieved_ids.append(doc_id)
            retrieved_docs.append(doc_text)
        
        return {
            "source": source,
            "retrieved_ids": retrieved_ids,
            "retrieved_docs": retrieved_docs,
            "confidence": 0.9 if retrieved_ids else 0.0
        }
    
    def _format_doc_text(self, item: Dict) -> str:
        """Format compliance item as readable text"""
        if "element_name" in item:
            return f"Compliance Issue: {item.get('element_name')} - {item.get('issue', '')}"
        elif "regulation_clause" in item:
            return f"Gap Analysis: {item.get('regulation_clause')} vs {item.get('requirement', '')}"
        else:
            return str(item)
    
    def _retrieve_from_vector_store(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve from L1-L5 only FAISS vector store"""
        if self.index is None:
            return {"source": "error", "retrieved_ids": [], "retrieved_docs": [], "confidence": 0.0}
        
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        retrieved_ids = []
        retrieved_docs = []
        
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                chunk = self.text_chunks[idx]
                retrieved_docs.append(chunk)
                retrieved_ids.append(f"chunk_{idx}")
        
        return {
            "source": "l1l5_vector_store",
            "retrieved_ids": retrieved_ids,
            "retrieved_docs": retrieved_docs,
            "confidence": 0.85 if retrieved_ids else 0.0,
            "distances": distances[0].tolist() if len(distances) > 0 else []
        }
    
    def route_query(self, query: str, top_k: int = 5, mode: str = "auto") -> Dict[str, Any]:
        """
        Route query to appropriate data source.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            mode: 
                - "compliance_only": Force JSON only (Stage 1)
                - "semantic_only": Force Vector DB only (L1-L5)
                - "auto": Intent-based routing (Stage 2)
        
        Returns:
            Dictionary with source, retrieved_ids, retrieved_docs, confidence
        """
        # Stage 1: Force compliance-only mode
        if mode == "compliance_only":
            return self._retrieve_from_compliance(query, top_k)
        
        # Stage 3: Force semantic-only mode (L1-L5 only)
        if mode == "semantic_only":
            return self._retrieve_from_vector_store(query, top_k)
        
        # Stage 2: Auto mode - Intent-based routing
        # Compliance queries → JSON, others → Vector DB (L1-L5)
        if self._is_compliance_query(query):
            return self._retrieve_from_compliance(query, top_k)
        else:
            return self._retrieve_from_vector_store(query, top_k)
    
    def get_all_docs(self) -> List[str]:
        """Get all document texts for ground truth"""
        return self.text_chunks
    
    def get_doc_ids(self) -> List[str]:
        """Get all document IDs for ground truth"""
        return [f"chunk_{i}" for i in range(len(self.text_chunks))]