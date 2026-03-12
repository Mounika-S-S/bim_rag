# src/rag/unified_vector_store.py
"""
Unified vector store combining all layers
FIXED: Rich L1 indexing with all properties
Based on entity-centric retrieval research [SIGIR'23]
"""
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.json_storage import JSONStorage
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UnifiedVectorStore:
    """
    Unified vector store with hybrid search capabilities
    Supports L1, L2, L4, L5, compliance, and comparison layers
    
    Entity-centric design based on SIGIR'23 research:
    - Each entity (element) gets multiple searchable chunks
    - Property-level indexing for fine-grained retrieval
    - Entity name boosting for exact matches
    """
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []
        self.chunk_metadata = []
        self.bm25 = None
        self.bm25_corpus = []
        self.entity_name_index = {}  # Quick lookup by name
    
    def build_from_project(self, project_id):
        """
        Build unified knowledge base from ALL layers
        Enhanced L1 indexing with all properties
        """
        print(f"\n📚 Building unified vector store for project: {project_id}")
        
        # Load all layers
        l1 = JSONStorage.load(project_id, "L1_ifc.json") or []
        l2 = JSONStorage.load(project_id, "L2_product.json") or []
        l4 = JSONStorage.load(project_id, "L4_regulation.json") or []
        l5 = JSONStorage.load(project_id, "L5_requirement.json") or []
        mismatch = JSONStorage.load(project_id, "mismatch.json") or []
        l4_l5_comparisons = JSONStorage.load(project_id, "l4_l5_mismatch.json") or []
        
        print(f"   L1: {len(l1)} elements")
        print(f"   L2: {len(l2)} products")
        print(f"   L4: {len(l4)} regulations")
        print(f"   L5: {len(l5)} requirements")
        print(f"   Mismatch: {len(mismatch)} issues")
        print(f"   L4-L5 Comparisons: {len(l4_l5_comparisons)} comparisons")
        
        # Clear previous chunks
        self.text_chunks = []
        self.chunk_metadata = []
        self.entity_name_index = {}
        
        # -----------------------------
        # L1 IFC Elements - ENHANCED with all properties
        # Based on entity-centric retrieval [SIGIR'23]
        # -----------------------------
        l1_count = 0
        for element in l1:
            props = element.get('properties', {})
            name = props.get('Name', 'Unknown')
            etype = element.get('entity_type', 'Element')
            element_id = element.get('id', 'unknown')
            
            # Store in name index for quick lookup
            if name != 'Unknown':
                self.entity_name_index[name] = {
                    'id': element_id,
                    'type': etype,
                    'chunk_index': len(self.text_chunks)  # Will be updated
                }
            
            # === CHUNK 1: Main element description ===
            main_chunk = f"[IFC Element] {etype} '{name}' (ID: {element_id})"
            self.text_chunks.append(main_chunk)
            self.chunk_metadata.append({
                "layer": "L1",
                "type": "element_main",
                "id": element_id,
                "name": name,
                "entity_type": etype,
                "all_props": str(props)[:500]  # Store for search
            })
            
            # === CHUNK 2: All properties as searchable text ===
            if props:
                prop_parts = [f"[Properties] {etype} '{name}'"]
                for key, value in props.items():
                    if value and key not in ['id', 'GlobalId', 'OwnerHistory']:
                        prop_parts.append(f"{key}: {value}")
                
                prop_chunk = " | ".join(prop_parts)
                self.text_chunks.append(prop_chunk)
                self.chunk_metadata.append({
                    "layer": "L1",
                    "type": "element_properties",
                    "id": element_id,
                    "name": name,
                    "entity_type": etype,
                    "properties": props
                })
            
            # === CHUNK 3: Name-only for exact matching ===
            if name != 'Unknown':
                name_chunk = f"[Element Name] {name} is a {etype}"
                self.text_chunks.append(name_chunk)
                self.chunk_metadata.append({
                    "layer": "L1",
                    "type": "element_name",
                    "id": element_id,
                    "name": name,
                    "entity_type": etype
                })
            
            l1_count += 1
        
        print(f"   Created {l1_count * 3} L1 chunks (main + properties + name)")
        
        # -----------------------------
        # L2 Products
        # -----------------------------
        for product in l2:
            props = product.get('properties', {})
            name = props.get('Product_Name', 'Unknown')
            mfr = props.get('Manufacturer', 'Unknown')
            
            chunk = f"[Product] {name} from {mfr}"
            # Add key specifications
            for key in ['Fire_Rating_Hours', 'Unit_Cost_INR', 'Thickness_mm']:
                if key in props:
                    chunk += f" | {key}: {props[key]}"
            
            self.text_chunks.append(chunk)
            self.chunk_metadata.append({
                "layer": "L2",
                "type": "product",
                "id": product.get('id'),
                "name": name,
                "manufacturer": mfr,
                "properties": props
            })
        
        # -----------------------------
        # L4 Regulations
        # -----------------------------
        for rule in l4:
            props = rule.get('properties', {})
            text = props.get('text', '')
            if text:
                # Clean text - remove very long definitions
                if len(text) > 1000:
                    text = text[:500] + "..."
                
                chunk = f"[Regulation] {props.get('rule_type', 'General')}: {text}"
                self.text_chunks.append(chunk)
                self.chunk_metadata.append({
                    "layer": "L4",
                    "type": "regulation",
                    "id": rule.get('id'),
                    "rule_type": props.get('rule_type', 'General'),
                    "threshold": props.get('threshold_value'),
                    "unit": props.get('unit')
                })
        
        # -----------------------------
        # L5 Requirements
        # -----------------------------
        for req in l5:
            props = req.get('properties', {})
            req_id = props.get('RequirementID', 'Unknown')
            desc = props.get('Description', '')
            rule = f"{props.get('Property', '')} {props.get('Operator', '')} {props.get('RequiredValue', '')} {props.get('Unit', '')}"
            
            chunk = f"[Project Requirement] {req_id}: {desc} (Rule: {rule}) Priority: {props.get('Priority', 'Medium')}"
            self.text_chunks.append(chunk)
            self.chunk_metadata.append({
                "layer": "L5",
                "type": "requirement",
                "id": req.get('id'),
                "requirement_id": req_id,
                "description": desc,
                "rule": rule,
                "priority": props.get('Priority', 'Medium')
            })
        
        # -----------------------------
        # Mismatch Records (L1+L2+L4/L5)
        # -----------------------------
        for issue in mismatch:
            element = issue.get('element_name', 'Unknown')
            problem = issue.get('issue', '')
            
            chunk = f"[Compliance Issue] {element} is NON-COMPLIANT: {problem}"
            self.text_chunks.append(chunk)
            self.chunk_metadata.append({
                "layer": "compliance",
                "type": "issue",
                "element_id": issue.get('element_id'),
                "requirement_id": issue.get('requirement_id'),
                "issue": problem
            })
        
        # -----------------------------
        # L4-L5 Comparisons
        # -----------------------------
        for comp in l4_l5_comparisons:
            relationship = comp.get('relationship', 'unknown')
            gap_desc = comp.get('gap_description', '')
            l4_rule = comp.get('l4_rule', {})
            l5_req = comp.get('l5_requirement', {})
            
            l4_text = l4_rule.get('text', '')[:100] if l4_rule else ''
            l5_id = l5_req.get('requirement_id', '') if l5_req else 'None'
            
            if relationship == 'stricter':
                chunk = f"[Code Comparison] 🔼 PROJECT IS STRICTER THAN CODE: {gap_desc} | L4: {l4_text} | L5: {l5_id}"
            elif relationship == 'weaker':
                chunk = f"[Code Comparison] 🔽 PROJECT IS WEAKER THAN CODE: {gap_desc} | L4: {l4_text} | L5: {l5_id}"
            elif relationship == 'missing':
                chunk = f"[Code Comparison] ❌ CODE REQUIREMENT MISSING FROM PROJECT: {gap_desc} | L4: {l4_text}"
            elif relationship == 'equal':
                chunk = f"[Code Comparison] ✅ PROJECT MATCHES CODE: {gap_desc} | L4: {l4_text} | L5: {l5_id}"
            else:
                chunk = f"[Code Comparison] ⚠️ {comp.get('comparison', 'Unknown comparison')}"
            
            self.text_chunks.append(chunk)
            self.chunk_metadata.append({
                "layer": "comparison",
                "type": "l4_l5",
                "id": comp.get('id'),
                "comparison_id": comp.get('comparison_id'),
                "relationship": relationship,
                "l4_rule_id": l4_rule.get('id') if l4_rule else None,
                "l5_req_id": l5_req.get('id') if l5_req else None
            })
        
        print(f"\n✅ Created {len(self.text_chunks)} total chunks")
        
        # Create dense embeddings (semantic)
        embeddings = self.model.encode(self.text_chunks)
        embeddings = np.array(embeddings).astype("float32")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        # Create sparse index (keyword) using BM25
        self.bm25_corpus = [chunk.lower().split() for chunk in self.text_chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        print(f"✅ Built hybrid index with {self.index.ntotal} vectors")
        print(f"   Entity name index has {len(self.entity_name_index)} entries")
    
    def hybrid_search(self, query, k=5, alpha=0.5):
        """
        Hybrid search combining dense and sparse retrieval
        Enhanced with entity name boosting [TREC Entity Track]
        """
        if self.index is None or self.bm25 is None:
            print("❌ Index not built. Run build_from_project first.")
            return []
        
        query_lower = query.lower()
        
        # Check for exact entity name matches (highest boost)
        entity_boost = {}
        for name, info in self.entity_name_index.items():
            if name.lower() in query_lower:
                # Exact name match - boost by 50%
                entity_boost[info['chunk_index']] = 0.5
                # Also boost related chunks
                for i, meta in enumerate(self.chunk_metadata):
                    if meta.get('name') == name:
                        entity_boost[i] = 0.3
        
        # Dense retrieval (semantic)
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)
        
        dense_scores, dense_indices = self.index.search(query_embedding, k * 3)
        dense_scores = dense_scores[0]
        
        # Sparse retrieval (keyword)
        query_tokens = query_lower.split()
        sparse_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize sparse scores
        if max(sparse_scores) > 0:
            sparse_scores = sparse_scores / max(sparse_scores)
        
        # Combine scores with entity boosting
        combined_scores = {}
        
        # Add dense scores
        for i, idx in enumerate(dense_indices[0]):
            combined_scores[idx] = alpha * dense_scores[i]
        
        # Add sparse scores
        for idx, score in enumerate(sparse_scores):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * score
            elif score > 0:
                combined_scores[idx] = (1 - alpha) * score
        
        # Add entity boosts
        for idx, boost in entity_boost.items():
            if idx in combined_scores:
                combined_scores[idx] += boost
            else:
                combined_scores[idx] = boost
        
        # Get top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in sorted_results:
            results.append({
                "content": self.text_chunks[idx],
                "metadata": self.chunk_metadata[idx],
                "relevance_score": float(score)
            })
        
        return results
    
    def search_by_entity_name(self, entity_name, k=5):
        """Direct lookup by entity name (fast path)"""
        results = []
        for name, info in self.entity_name_index.items():
            if entity_name.lower() in name.lower():
                idx = info['chunk_index']
                results.append({
                    "content": self.text_chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "relevance_score": 1.0
                })
                if len(results) >= k:
                    break
        return results
    
    def search(self, query, k=5):
        """Legacy method - uses hybrid search with default alpha=0.5"""
        return self.hybrid_search(query, k, alpha=0.5)
    
    def save(self, path):
        """Save index and metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        
        with open(path + ".metadata", "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.text_chunks,
                "metadata": self.chunk_metadata,
                "entity_name_index": self.entity_name_index
            }, f, indent=2)
        
        print(f"✅ Saved unified index to {path}")
    
    def load(self, path):
        """Load index and metadata"""
        self.index = faiss.read_index(path)
        
        with open(path + ".metadata", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.text_chunks = data["chunks"]
            self.chunk_metadata = data["metadata"]
            self.entity_name_index = data.get("entity_name_index", {})
        
        # Rebuild BM25 index
        self.bm25_corpus = [chunk.lower().split() for chunk in self.text_chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        print(f"✅ Loaded unified index with {len(self.text_chunks)} chunks")
        print(f"   Entity name index has {len(self.entity_name_index)} entries")