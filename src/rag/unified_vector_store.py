import os
import json
import re

import faiss
import numpy as np

from src.core.json_storage import JSONStorage
from src.core.model_manager import model_manager


class UnifiedVectorStore:

    def __init__(self):

        # Use shared model manager to avoid repeated downloads
        self.model = model_manager.get_model("all-mpnet-base-v2")
        self.index = None
        self.text_chunks = []

    # --------------------------------------------------
    # Build unified knowledge base
    # --------------------------------------------------

    def build_from_project(self, project_id):

        knowledge_chunks = []

        # =========================
        # L1 IFC Elements
        # =========================
        l1 = JSONStorage.load(project_id, "L1_ifc.json")

        for element in l1:
            text = self._build_l1_chunk(element)
            if text:
                knowledge_chunks.append(text)

        # =========================
        # L2 Products
        # =========================
        l2 = JSONStorage.load(project_id, "L2_product.json")

        for product in l2:
            text = self._build_l2_chunk(product)
            if text:
                knowledge_chunks.append(text)

        # =========================
        # L3 Process Rules
        # =========================
        l3 = JSONStorage.load(project_id, "L3_process.json")

        for process in l3:
            props = process.get("properties", {})
            text = self._clean_text(props.get("text", ""))
            if self._is_high_value_text(text, "L3", max_len=900):
                clause = props.get("clause")
                source = props.get("source_document")
                prefix = "[Process Rule]"
                if clause:
                    prefix += f" Clause {clause}."
                if source:
                    prefix += f" Source = {source}."
                knowledge_chunks.append(f"{prefix} {text}".strip())

        # =========================
        # L4 Regulations
        # =========================
        l4 = JSONStorage.load(project_id, "L4_regulation.json")

        for idx, rule in enumerate(l4, start=1):
            text = self._clean_text(rule.get("text", ""))
            if not self._is_high_value_text(text, "L4", max_len=900):
                continue

            parts = [f"[Regulation] Rule #{idx}."]

            rule_type = rule.get("rule_type")
            operator = rule.get("comparison_operator")
            threshold = rule.get("threshold_value")
            unit = rule.get("unit")

            if rule_type:
                parts.append(f"Type = {rule_type}.")
            if operator and threshold is not None:
                parts.append(f"Constraint = {operator} {threshold} {unit or ''}".strip() + ".")

            parts.append(text)
            knowledge_chunks.append(" ".join(parts))

        # =========================
        # L5 Requirements / Rate table
        # =========================
        l5 = JSONStorage.load(project_id, "L5_requirement.json")

        for req in l5:
            props = req.get("properties", {})

            code = props.get("item_code", "")
            desc = self._clean_text(props.get("description", "") or props.get("text", ""))
            unit = props.get("unit", "")
            rate = props.get("rate", "")
            source = props.get("source_document", "")

            if not self._is_high_value_text(desc, "L5", max_len=600):
                continue

            text = f"[Requirement] {desc}."
            if code:
                text += f" Code = {code}."
            if unit:
                text += f" Unit = {unit}."
            if rate not in ("", None):
                text += f" Rate = {rate}."
            if source:
                text += f" Source = {source}."

            knowledge_chunks.append(text)

        # =========================
        # Unified Compliance Records (compliance.json)
        # Schema from UnifiedComplianceBuilder:
        #   record["compliance"]["status"]        → COMPLIANT / NON_COMPLIANT
        #   record["compliance"]["is_compliant"]  → bool
        #   record["compliance"]["layer_responsible"]
        #   record["compliance"]["actual_value"]  → {value, unit, field}
        #   record["compliance"]["required_value"]→ {operator, value, unit, field}
        #   record["compliance"]["reason"]        → human explanation
        #   record["compliance"]["why_required"]  → why the rule exists
        #   record["element"]  → {id, name, type}
        #   record["product"]  → {name, layer}
        #   record["source"]   → {rule_text, origin}
        #   record["layers_involved"] → list[str]
        # =========================
        compliance = JSONStorage.load(project_id, "compliance.json")

        for record in compliance:
            cid    = record.get("compliance_id", "")
            layers = ", ".join(record.get("layers_involved", []))

            # --- nested sub-objects ---
            element  = record.get("element") or {}
            product  = record.get("product") or {}
            comp     = record.get("compliance") or {}
            source   = record.get("source") or {}

            elem_name  = element.get("name", "")
            elem_type  = element.get("type", "")
            prod_name  = product.get("name", "")

            # ← FIXED: read from comp sub-dict, not top-level
            status      = comp.get("status", "")                   # COMPLIANT / NON_COMPLIANT
            is_compliant= comp.get("is_compliant", True)
            layer_resp  = comp.get("layer_responsible", "")
            act_val     = comp.get("actual_value") or {}
            req_val     = comp.get("required_value") or {}
            reason      = comp.get("reason", "")                   # ← was compliance_explanation
            why_req     = comp.get("why_required", "")             # ← was why_value_required

            source_text   = source.get("rule_text", "")[:300]     # ← was source_rule
            source_origin = source.get("origin", "")

            # Build rich, semantically dense text chunk
            verdict = "NON-COMPLIANT" if not is_compliant else "COMPLIANT"
            text = (
                f"[Compliance] ID:{cid} Status:{verdict} "
                f"Layers:{layers} LayerResponsible:{layer_resp}. "
            )
            if elem_name:
                text += f"Element:{elem_name} ({elem_type}). "
            if prod_name:
                text += f"Product:{prod_name}. "
            if reason:
                text += f"Reason:{reason}. "
            if act_val:
                text += (
                    f"ActualValue:{act_val.get('value','')} "
                    f"{act_val.get('unit','')} Field:{act_val.get('field','')}. "
                )
            if req_val:
                text += (
                    f"RequiredValue:{req_val.get('operator','')} "
                    f"{req_val.get('value','')} {req_val.get('unit','')} "
                    f"Field:{req_val.get('field','')}. "
                )
            if why_req:
                text += f"WhyRequired:{why_req[:300]} "
            if source_text:
                text += f"SourceRule({source_origin}):{source_text}"

            if len(text.strip()) > 20:
                knowledge_chunks.append(text.strip())

        # =========================
        # L1-L2-L3 inference
        # =========================
        l123 = JSONStorage.load(project_id, "l123_inference.json")

        for r in l123:

            element = r.get("element_name", "")
            product = r.get("product", "")
            rule = self._clean_text(r.get("process_rule", ""))

            if self._is_high_value_text(rule, "L123", max_len=700):
                text = f"[L123 Inference] {element} uses {product}. Process rule = {rule}"
                knowledge_chunks.append(text)

        

        # =========================
        # L1-L2-L5 inference
        # =========================
        l125 = JSONStorage.load(project_id, "l125_inference.json")
        for r in l125:

            element = r.get("element_name", "")
            product = r.get("product", "")
            req = self._clean_text(r.get("requirement", ""))

            if self._is_high_value_text(req, "L125", max_len=700):
                text = f"[L125 Inference] {element} uses {product}. Requirement = {req}"
                knowledge_chunks.append(text)

        # =========================
        # L4-L5 inference
        # =========================
        l45 = JSONStorage.load(project_id, "l45_inference.json")

        for r in l45:

            reg = self._clean_text(r.get("regulation_clause", ""))
            req = self._clean_text(r.get("requirement", ""))

            if self._is_high_value_text(reg, "L45", max_len=700) and self._is_high_value_text(req, "L45", max_len=700):
                text = f"[L45 Inference] Regulation = {reg}. Requirement = {req}"
                knowledge_chunks.append(text)

        # =========================
        # Clean and dedupe chunks
        # =========================
        knowledge_chunks = self._dedupe_chunks(knowledge_chunks)

        if not knowledge_chunks:
            print("No knowledge chunks found.")
            return

        self.text_chunks = knowledge_chunks

        # =========================
        # Generate embeddings
        # =========================
        embeddings = self.model.encode(knowledge_chunks)

        embeddings = np.array(embeddings).astype("float32")

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Use HNSW index for better performance on larger datasets
        # M=32 provides good balance of speed vs accuracy
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128

        self.index.add(embeddings)

        print(f"Vector store built with {len(knowledge_chunks)} chunks.")

    # --------------------------------------------------
    # Save index
    # --------------------------------------------------

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, path)

        with open(path + ".texts", "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f, ensure_ascii=False)

        print("Vector store saved.")

    # --------------------------------------------------
    # Load index
    # --------------------------------------------------

    def load(self, path):

        self.index = faiss.read_index(path)

        expected_dim = self.model.get_sentence_embedding_dimension()
        if hasattr(self.index, "d") and self.index.d != expected_dim:
            raise ValueError(
                f"Embedded dimension mismatch: index is {self.index.d}, "
                f"model is {expected_dim}. Please rebuild vector store (option 11)."
            )

        with open(path + ".texts", "r", encoding="utf-8") as f:
            self.text_chunks = json.load(f)

        print("Vector store loaded.")

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(self, query, k=5):

        query_embedding = self.model.encode([query])

        query_embedding = np.array(query_embedding).astype("float32")

        faiss.normalize_L2(query_embedding)

        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = max(k * 2, 64)

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:

            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results


    # --------------------------------------------------
    # Chunk builders
    # --------------------------------------------------

    def _build_l1_chunk(self, element):
        props = element.get("properties", {})
        name = self._clean_text(str(props.get("Name", "") or ""))
        object_type = self._clean_text(str(props.get("ObjectType", "") or ""))

        if not name or name == "-":
            return None

        important_keys = [
            "Catalog reference",
            "Question 2",
            "Answer 1",
            "Answer 2",
            "Length",
            "Width",
            "Height",
            "FireRating",
            "Unit",
            "NOI_UUID",
            "Allright_Comp_ID",
        ]

        parts = [f"[IFC Element] Name = {name}."]

        entity_type = element.get("entity_type")
        if entity_type:
            parts.append(f"Entity = {entity_type}.")
        if object_type and object_type.lower() != "none":
            parts.append(f"ObjectType = {object_type}.")

        for key in important_keys:
            value = props.get(key)
            if self._should_keep_property(key, value):
                parts.append(f"{key} = {value}.")

        return " ".join(parts)

    def _build_l2_chunk(self, product):
        props = product.get("properties", {})
        source = props.get("source_document", "")

        product_name = (
            props.get("Product_Name")
            or props.get("product_name")
            or props.get("name")
            or ""
        )
        product_name = self._clean_text(str(product_name))

        if product_name:
            parts = [f"[Product] Name = {product_name}."]
            for key, value in props.items():
                if key in {"Product_Name", "product_name", "name"}:
                    continue
                if self._should_keep_property(key, value):
                    parts.append(f"{key} = {value}.")
            return " ".join(parts)

        text = self._clean_text(str(props.get("text", "") or ""))
        if not self._is_high_value_text(text, "L2", max_len=400):
            return None

        chunk = f"[Product Evidence] {text}."
        if props.get("numeric_value") not in (None, ""):
            chunk += f" Numeric value = {props.get('numeric_value')}."
        if source:
            chunk += f" Source = {source}."
        return chunk

    def _should_keep_property(self, key, value):
        if value in (None, "", "None"):
            return False

        key_lower = str(key).lower()
        if key_lower.startswith("v") and value == 0.0:
            return False

        text = self._clean_text(str(value))
        if not text:
            return False

        return True

    def _clean_text(self, text):
        if not text:
            return ""

        text = str(text).replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _dedupe_chunks(self, chunks):
        unique_chunks = []
        seen = set()

        for chunk in chunks:
            cleaned = self._clean_text(chunk)
            if len(cleaned) < 20:
                continue

            normalized = re.sub(r"\s+", " ", cleaned.lower())
            if normalized in seen:
                continue

            seen.add(normalized)
            unique_chunks.append(cleaned)

        return unique_chunks

    def _is_high_value_text(self, text, layer, max_len):
        text = self._clean_text(text)
        if len(text) < 20 or len(text) > max_len:
            return False

        lowered = text.lower()

        noisy_markers = [
            "bureau of indian standards",
            "supplied by book supply bureau",
            "under the license from bis",
            "table of contents",
            "national building code of india part 3 development control rules",
            "7102-21-13otpu",
        ]

        if any(marker in lowered for marker in noisy_markers):
            return False

        if layer == "L2":
            weak_markers = [
                "american concrete institute",
                "cement and concrete association",
                "handbook on concrete engineering",
            ]
            if any(marker in lowered for marker in weak_markers):
                return False

        if layer in {"L3", "L4", "L123", "L125", "L45"}:
            if lowered.count("figure") > 1 or lowered.count("table") > 2:
                return False

        return True
