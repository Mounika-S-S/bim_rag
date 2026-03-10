import os
from src.core.schema import create_layer_record
from src.ingestion.base_pdf_cleaner import BasePDFCleaner
from src.l3.clause_segmenter import ClauseSegmenter
from src.l3.hybrid_process_extractor import HybridProcessExtractor


class L3Pipeline:

    def __init__(self):

        self.cleaner = BasePDFCleaner()
        self.segmenter = ClauseSegmenter()
        self.extractor = HybridProcessExtractor()

    def parse(self, pdf_path):

        text = self.cleaner.extract_text(pdf_path)

        clauses = self.segmenter.segment(text)

        extracted_rules = self.extractor.extract(clauses)

        records = []

        for i, rule in enumerate(extracted_rules):

            record = create_layer_record(
                record_id=f"L3_{i+1}",
                entity_type="ProcessRule",
                layer="L3",
                category="InformationManagementProcess",
                properties={
                    "text": rule["text"],
                    "source_document": os.path.basename(pdf_path)
                }
            )

            records.append(record)

        return records