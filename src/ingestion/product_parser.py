import json
from typing import List, Dict, Any


class ProductParser:
    """
    Parses product data (JSON for now)
    and converts to standardized internal format.
    """

    STRUCTURAL_KEYS = {"element_id", "id", "entity", "type", "layer"}

    def parse_json(self, file_path: str) -> List[Dict[str, Any]]:

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        elements = []

        for item in data:
            if not isinstance(item, dict):
                continue

            element_id = item.get("element_id") or item.get("id")

            if not element_id:
                continue

            attributes = {
                key: value
                for key, value in item.items()
                if key not in self.STRUCTURAL_KEYS
            }

            elements.append({
                "element_id": str(element_id),
                "entity": "product",
                "layer": "L2",
                "attributes": attributes
            })

        return elements