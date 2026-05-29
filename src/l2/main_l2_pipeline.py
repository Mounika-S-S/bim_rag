import os
from src.core.schema import create_layer_record
from src.ingestion.base_pdf_cleaner import BasePDFCleaner
from src.l2.product_segmenter import ProductSegmenter
from src.l2.hybrid_product_extractor import HybridProductExtractor


class L2Pipeline:

    def __init__(self):

        self.cleaner = BasePDFCleaner()
        self.segmenter = ProductSegmenter()
        self.extractor = HybridProductExtractor()

    def parse(self, pdf_path):

        text = self.cleaner.extract_text(pdf_path)

        segments = self.segmenter.segment(text)

        extracted = self.extractor.extract(segments)

        records = []

        for i, p in enumerate(extracted):

            record = create_layer_record(
                record_id=f"L2_{i}",
                entity_type="Product",
                layer="L2",
                category="ConstructionMaterial",
                properties={
                    "text": p["property_text"],
                    "numeric_value": p["numeric_value"],
                    "source_document": os.path.basename(pdf_path)
                }
            )

            records.append(record)

        return records