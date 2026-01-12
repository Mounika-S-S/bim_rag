import json
import chromadb
from chromadb.utils import embedding_functions

DATA_FILES = {
    "mismatches": "data/output/mismatches.json",
    "rules": "data/output/rules.json",
    "products": "data/output/products.json",
    "documents": "data/output/documents.json",
    "regulations": "data/output/regulations.json",
}

DB_PATH = "data/vector_db"
BATCH_SIZE = 10


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # ✅ PERSISTENT CLIENT (FIX)
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name="bim_rag",
        embedding_function=embedding_fn
    )

    all_docs, all_ids, all_meta = [], [], []
    doc_id = 0

    print("Collecting documents for embedding...")

    for source, path in DATA_FILES.items():
        for item in load_json(path):
            all_docs.append(json.dumps(item))
            all_ids.append(f"{source}_{doc_id}")
            all_meta.append({"source": source})
            doc_id += 1

    print(f"Total documents to embed: {doc_id}")
    print("Embedding in batches...")

    for i in range(0, len(all_docs), BATCH_SIZE):
        collection.add(
            documents=all_docs[i:i+BATCH_SIZE],
            ids=all_ids[i:i+BATCH_SIZE],
            metadatas=all_meta[i:i+BATCH_SIZE],
        )
        print(f"Embedded {i + BATCH_SIZE} / {len(all_docs)}")

    print("PHASE 4A COMPLETE — VECTOR DB BUILT AND PERSISTED")


if __name__ == "__main__":
    main()
