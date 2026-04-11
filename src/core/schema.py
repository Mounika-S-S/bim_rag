# src/core/schema.py

def create_layer_record(
    record_id,
    entity_type,
    layer,
    category,
    properties,
    element_type_normalized=None,
):
    record = {
        "id": record_id,
        "entity_type": entity_type,
        "layer": layer,
        "category": category,
        "properties": properties,
    }
    if element_type_normalized:
        record["element_type_normalized"] = element_type_normalized
    return record