from src.ingestion.chunk_builder import ChunkBuilder
from src.embedding.vector_store import VectorStore


class IngestionPipeline:

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.vector_store = VectorStore(project_id)

    def ingest_elements(self, elements):

        all_documents = []
        all_metadatas = []
        all_ids = []

        for element in elements:

            base_metadata = {
                "layer": element["layer"],
                "entity": element["entity"],
                "element_id": element["element_id"]
            }

            chunks = ChunkBuilder.flatten_json_to_chunks(
                element["attributes"],
                base_metadata
            )

            for i, chunk in enumerate(chunks):
                doc_id = f"{element['element_id']}_{i}"

                all_documents.append(chunk["text"])
                all_metadatas.append(chunk["metadata"])
                all_ids.append(doc_id)

        if all_documents:
            self.vector_store.add_documents(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )