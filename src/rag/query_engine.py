import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "data/vector_db"


def main():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name="bim_rag",
        embedding_function=embedding_fn
    )

    print("BIM RAG Query Engine Ready")
    print("Type a question (or 'exit'):")

    while True:
        query = input("\n> ")
        if query.lower() == "exit":
            break

        results = collection.query(
            query_texts=[query],
            n_results=5,
            where={"source": "mismatches"}
        )

        print("\n--- Fire Safety Violations ---")
        for doc in results["documents"][0]:
            print(doc)
            print("-----")


if __name__ == "__main__":
    main()
