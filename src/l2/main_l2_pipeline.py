import os
from src.core.schema import create_layer_record
from src.l2.structured_product_parser import StructuredProductParser


class L2Pipeline:

    def __init__(self):
        self.parser = StructuredProductParser()

    def parse(self, file_path: str) -> list:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls", ".csv"):
            raw_records = self.parser.parse_excel(file_path)
        else:
            raw_records = self.parser.parse_pdf(file_path)

        records = []
        for i, r in enumerate(raw_records):
            record = create_layer_record(
                record_id=f"L2_{i}",
                entity_type="Product",
                layer="L2",
                category="ConstructionMaterial",
                properties=r["properties"],
                element_type_normalized=r.get("element_type_normalized"),
            )
            records.append(record)

        return records