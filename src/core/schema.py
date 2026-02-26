# src/core/schema.py

def create_layer_record(
    record_id,
    entity_type,
    layer,
    category,
    properties
):
    return {
        "id": record_id,
        "entity_type": entity_type,
        "layer": layer,
        "category": category,
        "properties": properties
    }