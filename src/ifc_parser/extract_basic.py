def extract_basic_info(element):
    """
    Extracts basic identifying information from an IFC element.
    """
    return {
        "global_id": element.GlobalId,
        "name": element.Name,
        "type": element.is_a()
    }
