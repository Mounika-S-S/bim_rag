class ChunkBuilder:

    IGNORE_KEYS = {"element_id", "id", "guid"}

    @staticmethod
    def flatten_json_to_chunks(data, base_metadata, parent_key=""):
        chunks = []

        for key, value in data.items():

            if key in ChunkBuilder.IGNORE_KEYS:
                continue

            full_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                chunks.extend(
                    ChunkBuilder.flatten_json_to_chunks(
                        value,
                        base_metadata,
                        parent_key=full_key
                    )
                )

            else:
                entity = base_metadata.get("entity", "Element")
                element_id = base_metadata.get("element_id", "")

                readable_property = full_key.replace("_", " ").replace(".", " > ")

                text = f"{entity.capitalize()} {element_id} has {readable_property} value {value}."

                metadata = base_metadata.copy()
                metadata["property_path"] = full_key

                chunks.append({
                    "text": text,
                    "metadata": metadata
                })

        return chunks