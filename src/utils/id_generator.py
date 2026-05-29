# src/utils/id_generator.py

import uuid


def generate_id(prefix: str = None) -> str:
    """
    Generates a unique ID for layer records.
    
    Example:
        generate_id() → 'a8f9c2e4'
        generate_id("L2") → 'L2_a8f9c2e4'
    """

    unique_part = uuid.uuid4().hex[:8]

    if prefix:
        return f"{prefix}_{unique_part}"

    return unique_part