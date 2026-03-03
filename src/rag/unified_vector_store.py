# src/rag/unified_vector_store.py

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

        # -----------------------------
        # L4 Regulations
        # -----------------------------
        l4 = JSONStorage.load(project_id, "L4_regulation.json")

        for rule in l4:
            text = rule.get("text")
            if text and len(text.strip()) > 20:
                knowledge_chunks.append(
                    f"[Regulation] {text.strip()}"
                )

        # -----------------------------
        # L2 Products
        # -----------------------------
        l2 = JSONStorage.load(project_id, "L2_product.json")

        for product in l2:
            props = product.get("properties", {})
            product_name = props.get("Product_Name", "Unknown Product")

            description = f"[Product] {product_name}. "

            for key, value in props.items():
                description += f"{key} = {value}. "

            knowledge_chunks.append(description.strip())

        # -----------------------------
        # Mismatch Records
        # -----------------------------
        mismatch = JSONStorage.load(project_id, "mismatch.json")

        for issue in mismatch:

            element_id = issue.get("element_id", "")
            element_name = issue.get("element_name", "")
            element_type = issue.get("element_type", "")
            rule_text = issue.get("rule_text", "")
            product_value = issue.get("product_value", "")
            required = issue.get("required", "")
            unit = issue.get("unit", "")

            # Skip incomplete mismatch records
            if not rule_text or product_value == "" or required == "":
                continue

            description = (
                f"[Compliance Issue] The {element_type} '{element_name}' "
                f"is NON-COMPLIANT. "
                f"Rule: {rule_text}. "
                f"Provided value: {product_value} {unit}. "
                f"Required: {required} {unit}. "
                f"Element ID: {element_id}."
            )

            knowledge_chunks.append(description.strip())

        # Save chunks
        self.text_chunks = knowledge_chunks

        # Create embeddings
        embeddings = self.model.encode(knowledge_chunks)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Vector store built with {len(knowledge_chunks)} chunks.")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, path)

        with open(path + ".texts", "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f)

        print("Vector store saved.")

    # --------------------------------------------------
    # Load
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
            results.append(self.text_chunks[idx])

        return results