import os
import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    """
    Persistent vector store per project.
    """

    def __init__(self, project_id: str):

        self.project_id = project_id

        # Ensure base directory exists
        os.makedirs("vector_store", exist_ok=True)

        # Create persistent client for this project
        self.client = chromadb.PersistentClient(
            path=f"vector_store/{project_id}"
        )

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="bim_knowledge_base",
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents, metadatas, ids):
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text, n_results=5, where_filter=None):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

    def reset_collection(self):
        self.client.delete_collection("bim_knowledge_base")
        self.collection = self.client.get_or_create_collection(
            name="bim_knowledge_base",
            embedding_function=self.embedding_function
        )