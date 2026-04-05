"""
src/retrieval/deterministic_router.py
Router for compliance queries - routes to l124, l125, l45 JSON files
"""

import json
import os
from typing import List, Dict, Any


class DeterministicRouter:
    """
    Router for deterministic compliance queries.
    Routes queries to pre-computed inference JSON files:
    - l124_inference.json (L1-L2-L4 compliance)
    - l125_inference.json (L1-L2-L5 requirement checking)
    - l45_inference.json (L4-L5 gap analysis)
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.base_path = f"data/processed/{project_id}"
        
        # Load inference files
        self.l124_data = self._load_json("l124_inference.json")
        self.l125_data = self._load_json("l125_inference.json")
        self.l45_data = self._load_json("l45_inference.json")
        
        # Combined data for retrieval
        self.all_data = self.l124_data + self.l125_data + self.l45_data
    
    def _load_json(self, filename: str) -> List[Dict]:
        """Load JSON file if exists"""
        filepath = os.path.join(self.base_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def route_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Route query to appropriate compliance data
        
        Returns:
            {
                "source": "l124" / "l125" / "l45",
                "results": List[Dict],
                "retrieved_ids": List[str],
                "retrieved_docs": List[str],
                "confidence": float
            }
        """
        query_lower = query.lower()
        
        # Detect which inference file to use
        if any(word in query_lower for word in ["non-compliant", "non compliant", "violation", "l124"]):
            source = "l124"
            data = self.l124_data
            confidence = 0.9
        elif any(word in query_lower for word in ["requirement", "missing", "l125"]):
            source = "l125"
            data = self.l125_data
            confidence = 0.85
        elif any(word in query_lower for word in ["gap", "compare", "difference", "l45"]):
            source = "l45"
            data = self.l45_data
            confidence = 0.8
        else:
            # Default to all compliance data
            source = "all_compliance"
            data = self.all_data
            confidence = 0.7
        
        # Simple keyword matching for retrieval
        results = self._retrieve_relevant(query, data, top_k)
        
        return {
            "source": source,
            "results": results,
            "retrieved_ids": [r.get("element_id", f"doc_{i}") for i, r in enumerate(results)],
            "retrieved_docs": [self._format_doc_text(r) for r in results],
            "confidence": confidence
        }
    
    def _retrieve_relevant(self, query: str, data: List[Dict], top_k: int) -> List[Dict]:
        """Simple keyword-based retrieval from compliance data"""
        query_terms = set(query.lower().split())
        
        scored_results = []
        for item in data:
            # Combine all text fields for matching
            text = str(item).lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored_results.append((score, item))
        
        # Sort by score and return top_k
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_results[:top_k]]
    
    def _format_doc_text(self, item: Dict) -> str:
        """Format compliance item as readable text"""
        if "element_name" in item:
            return f"Compliance Issue: {item.get('element_name')} - {item.get('issue', '')}"
        elif "regulation_clause" in item:
            return f"Gap Analysis: {item.get('regulation_clause')} vs {item.get('requirement', '')}"
        else:
            return str(item)
    
    def get_all_docs(self) -> List[str]:
        """Get all document texts for ground truth"""
        return [self._format_doc_text(item) for item in self.all_data]
    
    def get_doc_ids(self) -> List[str]:
        """Get all document IDs for ground truth"""
        ids = []
        for item in self.all_data:
            if "element_id" in item:
                ids.append(item.get("element_id"))
            else:
                ids.append(f"doc_{len(ids)}")
        return ids