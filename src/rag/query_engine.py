import chromadb
from chromadb.utils import embedding_functions
from src.rag.query_router import route_query

DB_PATH = "data/vector_db"


def main():
    # Embedding function (same as used during indexing)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Persistent Chroma client
    client = chromadb.PersistentClient(path=DB_PATH)

    # Load or create collection
    collection = client.get_or_create_collection(
        name="bim_rag",
        embedding_function=embedding_fn
    )

    print("BIM RAG Query Engine Ready")
    print("Type a question (or 'exit'):")

    while True:
        query = input("\n> ").strip()
        if query.lower() == "exit":
            break

        # 🔹 Route query to relevant sources
        sources = route_query(query)

        # 🔹 Query vector DB
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where={"source": {"$in": sources}}
        )

        docs = results.get("documents", [[]])[0]

        # 🔹 Handle no data case
        if not docs:
            print("\nNo relevant data found for this question.")
            continue

        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(docs, start=1):
            print(f"\n[{i}]")
            print(doc)
            print("-" * 40)


if __name__ == "__main__":
    main()
