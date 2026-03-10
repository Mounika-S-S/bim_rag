import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.json_storage import JSONStorage


class UnifiedVectorStore:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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

            props = element.get("properties", {})
            name = props.get("Name", "Unknown Element")

            text = f"[IFC Element] {name}. "

            for k, v in props.items():
                text += f"{k} = {v}. "

            knowledge_chunks.append(text.strip())

        # =========================
        # L2 Products
        # =========================
        l2 = JSONStorage.load(project_id, "L2_product.json")

        for product in l2:

            props = product.get("properties", {})

            name = props.get("Product_Name", "Unknown Product")

            text = f"[Product] {name}. "

            for k, v in props.items():
                text += f"{k} = {v}. "

            knowledge_chunks.append(text.strip())

        # =========================
        # L3 Process Rules
        # =========================
        l3 = JSONStorage.load(project_id, "L3_process.json")

        for process in l3:

            text = process.get("properties", {}).get("text", "")

            if text and len(text) > 20:
                knowledge_chunks.append(f"[Process Rule] {text}")

        # =========================
        # L4 Regulations
        # =========================
        l4 = JSONStorage.load(project_id, "L4_regulation.json")

        for rule in l4:

            text = rule.get("properties", {}).get("text", "")

            if text and len(text) > 20:
                knowledge_chunks.append(f"[Regulation] {text}")

        # =========================
        # L5 Requirements / Rate table
        # =========================
        l5 = JSONStorage.load(project_id, "L5_requirement.json")

        for req in l5:

            props = req.get("properties", {})

            code = props.get("item_code", "")
            desc = props.get("description", "")
            unit = props.get("unit", "")
            rate = props.get("rate", "")

            if desc:
                text = f"[Requirement] {desc}. Code: {code}. Unit: {unit}. Rate: {rate}."
                knowledge_chunks.append(text)

        # ==========================================
        # L1-L2-L4 inference
        # ==========================================

        l124 = JSONStorage.load(project_id, "l124_inference.json")

        if l124:

            for issue in l124:

                element_name = issue.get("element_name", "")
                element_type = issue.get("element_type", "")
                rule_text = issue.get("rule_text", "")
                product_value = issue.get("product_value", "")
                required = issue.get("required", "")
                unit = issue.get("unit", "")

                if not rule_text:
                    continue

                text = (
                    f"[L124 Inference] {element_type} '{element_name}' "
                    f"is NON-COMPLIANT. Rule: {rule_text}. "
                    f"Provided: {product_value} {unit}. "
                    f"Required: {required} {unit}."
                )

                knowledge_chunks.append(text)

        # =========================
        # L1-L2-L3 inference
        # =========================
        l123 = JSONStorage.load(project_id, "l123_inference.json")

        for r in l123:

            element = r.get("element_name", "")
            product = r.get("product", "")
            rule = r.get("process_rule", "")

            text = f"[L123 Inference] {element} uses {product}. Process rule: {rule}"

            knowledge_chunks.append(text)

        # =========================
        # L1-L2-L5 inference
        # =========================
        l125 = JSONStorage.load(project_id, "l125_inference.json")

        for r in l125:

            element = r.get("element_name", "")
            product = r.get("product", "")
            req = r.get("requirement", "")

            text = f"[L125 Inference] {element} uses {product}. Requirement: {req}"

            knowledge_chunks.append(text)

        # =========================
        # L4-L5 inference
        # =========================
        l45 = JSONStorage.load(project_id, "l45_inference.json")

        for r in l45:

            reg = r.get("regulation_clause", "")
            req = r.get("requirement", "")

            text = f"[L45 Inference] Regulation: {reg}. Requirement: {req}"

            knowledge_chunks.append(text)

        # =========================
        # Clean empty chunks
        # =========================
        knowledge_chunks = [
            c for c in knowledge_chunks if c and len(c.strip()) > 20
        ]

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

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"Vector store built with {len(knowledge_chunks)} chunks.")

    # --------------------------------------------------
    # Save index
    # --------------------------------------------------

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, path)

        with open(path + ".texts", "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f)

        print("Vector store saved.")

    # --------------------------------------------------
    # Load index
    # --------------------------------------------------

    def load(self, path):

        self.index = faiss.read_index(path)

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

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:

            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results