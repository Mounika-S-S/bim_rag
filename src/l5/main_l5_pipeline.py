import os
from src.core.schema import create_layer_record
from src.l5.clause_segmenter import ClauseSegmenter
from src.utils.id_generator import generate_id


class L5Pipeline:

    def __init__(self):
        self.segmenter = ClauseSegmenter()

    def parse(self, pdf_path):

        rows = self.segmenter.segment(pdf_path)

        records = []

        for r in rows:

            record = create_layer_record(
                record_id=generate_id("L5"),
                entity_type="Requirement",
                layer="L5",
                category="ConstructionRequirement",
                properties={
                    "item_code": r["code"],
                    "description": r["description"],
                    "unit": r["unit"],
                    "rate": r["rate"],
                    "source_document": os.path.basename(pdf_path)
                }
            )

            records.append(record)

        return records