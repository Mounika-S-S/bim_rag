import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.json_storage import JSONStorage
from src.core.model_manager import model_manager


class UnifiedVectorStore:

    def __init__(self):
        self.model = model_manager.get_model("all-mpnet-base-v2")
        self.index = None
        self.text_chunks = []
        self.metadata = []   # parallel list: {layer, element_id, element_type, property}

    # ------------------------------------------------------------------
    def build_from_project(self, project_id):
        knowledge_chunks = []
        meta = []

        # ── L1 IFC Elements ──────────────────────────────────────────
        for element in JSONStorage.load(project_id, "L1_ifc.json"):
            props = element.get("properties", {})
            name = props.get("Name", "Unknown")
            etype = element.get("element_type_normalized", element.get("entity_type", "element"))
            eid = element.get("id", "")

            text = f"[IFC Element] Type={etype} Name={name}. "
            for k, v in props.items():
                if v is not None:
                    text += f"{k}={v}. "

            knowledge_chunks.append(text.strip())
            meta.append({"layer": "L1", "element_id": eid, "element_type": etype,
                          "element_name": name, "chunk_type": "element"})

        # ── L2 Products ───────────────────────────────────────────────
        for product in JSONStorage.load(project_id, "L2_product.json"):
            props = product.get("properties", {})
            etype = product.get("element_type_normalized") or props.get("ElementType", "product")
            eid = product.get("id", "")
            pname = props.get("ProductName", props.get("product_name", "Unknown Product"))

            text = f"[Product] Type={etype} Name={pname}. "
            for k, v in props.items():
                if v is not None:
                    text += f"{k}={v}. "

            knowledge_chunks.append(text.strip())
            meta.append({"layer": "L2", "element_id": eid, "element_type": etype,
                          "element_name": pname, "chunk_type": "product"})

        # ── L3 Process Rules ──────────────────────────────────────────
        for process in JSONStorage.load(project_id, "L3_process.json"):
            text = process.get("properties", {}).get("text", "")
            if text and len(text) > 20:
                knowledge_chunks.append(f"[Process Rule] {text}")
                meta.append({"layer": "L3", "element_id": process.get("id", ""),
                              "chunk_type": "process_rule"})

        # ── L4 Regulations ────────────────────────────────────────────
        for rule in JSONStorage.load(project_id, "L4_regulation.json"):
            text = rule.get("text", "") or rule.get("properties", {}).get("text", "")
            etypes = rule.get("element_types", [])
            if text and len(text) > 20:
                knowledge_chunks.append(f"[Regulation] {text}")
                meta.append({"layer": "L4", "element_id": "",
                              "element_types": etypes,
                              "rule_type": rule.get("rule_type", ""),
                              "chunk_type": "regulation"})

        # ── L5 Requirements ───────────────────────────────────────────
        for req in JSONStorage.load(project_id, "L5_requirement.json"):
            props = req.get("properties", {})
            desc = props.get("description", "")
            if desc:
                text = f"[Requirement] {desc}. Code: {props.get('item_code','')}. Unit: {props.get('unit','')}. Rate: {props.get('rate','')}."
                knowledge_chunks.append(text)
                meta.append({"layer": "L5", "element_id": req.get("id", ""), "chunk_type": "requirement"})

        # ── Compliance Results ─────────────────────────────────────────
        for issue in JSONStorage.load(project_id, "l124_inference.json"):
            status = issue.get("status", "")
            ename = issue.get("element_name", "")
            etype = issue.get("element_type", "")
            prop = issue.get("property", "")
            eff = issue.get("effective_value")
            req = issue.get("required_value")
            op = issue.get("operator", "")
            unit = issue.get("unit", "")
            rule_text = issue.get("source_rule", "")
            suggestion = issue.get("suggestion", "")

            if status == "COMPLIANT":
                text = (f"[Compliance] {etype} '{ename}' is COMPLIANT. "
                        f"Properties: {json.dumps(issue.get('properties_summary', {}))}")
            elif status == "NON_COMPLIANT":
                text = (f"[Compliance] {etype} '{ename}' is NON_COMPLIANT for {prop}. "
                        f"Actual={eff}{unit}, Required {op} {req}{unit}. "
                        f"Rule: {rule_text[:200]}. Suggestion: {suggestion}")
            elif status == "MISSING_PROPERTY":
                text = (f"[Compliance] {etype} '{ename}' — property '{prop}' NOT FOUND. "
                        f"Required {op} {req}{unit}. {suggestion}")
            else:
                continue

            knowledge_chunks.append(text)
            meta.append({"layer": "compliance", "element_type": etype,
                          "element_name": ename, "property": prop,
                          "status": status, "chunk_type": "compliance"})

        knowledge_chunks = [c for c in knowledge_chunks if c and len(c.strip()) > 20]
        self.metadata = meta[:len(knowledge_chunks)]

        if not knowledge_chunks:
            print("No knowledge chunks found.")
            return

        self.text_chunks = knowledge_chunks
        embeddings = np.array(self.model.encode(knowledge_chunks)).astype("float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128
        self.index.add(embeddings)
        print(f"Vector store built with {len(knowledge_chunks)} chunks.")

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        with open(path + ".texts", "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f)
        with open(path + ".meta", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)
        print("Vector store saved.")

    def load(self, path):
        self.index = faiss.read_index(path)
        expected_dim = self.model.get_sentence_embedding_dimension()
        if hasattr(self.index, "d") and self.index.d != expected_dim:
            raise ValueError(
                f"Dimension mismatch: index={self.index.d}, model={expected_dim}. Rebuild vector store."
            )
        with open(path + ".texts", "r", encoding="utf-8") as f:
            self.text_chunks = json.load(f)
        meta_path = path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = [{} for _ in self.text_chunks]
        print("Vector store loaded.")

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[str]:
        qemb = np.array(self.model.encode([query])).astype("float32")
        faiss.normalize_L2(qemb)
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = max(k * 2, 64)
        distances, indices = self.index.search(qemb, k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

    def search_with_metadata(self, query: str, k: int = 5,
                             filter_layer: str = None,
                             filter_element_type: str = None,
                             filter_status: str = None) -> list[dict]:
        """Return chunks WITH metadata, with optional filters."""
        qemb = np.array(self.model.encode([query])).astype("float32")
        faiss.normalize_L2(qemb)
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = max(k * 10, 128)
        distances, indices = self.index.search(qemb, min(k * 10, len(self.text_chunks)))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.text_chunks):
                continue
            m = self.metadata[idx] if idx < len(self.metadata) else {}
            if filter_layer and m.get("layer") != filter_layer:
                continue
            if filter_element_type:
                et = m.get("element_type", "").lower()
                if filter_element_type.lower() not in et:
                    continue
            if filter_status and m.get("status") != filter_status:
                continue
            results.append({"text": self.text_chunks[idx], "meta": m, "score": float(distances[0][i])})
            if len(results) >= k:
                break

        return results

    def get_all_element_types(self) -> list[str]:
        """Returns all unique element types in the store (for 'not found' responses)."""
        types = set()
        for m in self.metadata:
            et = m.get("element_type")
            if et:
                types.add(et)
        return sorted(types)

    def get_all_properties_for_type(self, element_type: str) -> list[str]:
        """Returns all property names recorded for a given element type."""
        props = set()
        query_type = element_type.lower()
        for i, m in enumerate(self.metadata):
            et = m.get("element_type", "").lower()
            if query_type in et:
                # parse chunk text for property keys
                text = self.text_chunks[i] if i < len(self.text_chunks) else ""
                for part in text.split(". "):
                    if "=" in part:
                        key = part.split("=")[0].strip().lstrip("[").strip()
                        if key and len(key) < 40:
                            props.add(key)
        return sorted(props)