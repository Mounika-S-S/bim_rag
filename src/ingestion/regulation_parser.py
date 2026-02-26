import os
import re
from PyPDF2 import PdfReader
from src.core.schema import create_layer_record


class RegulationParser:

    def __init__(self):
        pass

    def parse_pdf(self, file_path):

        reader = PdfReader(file_path)

        records = []
        file_name = os.path.basename(file_path)
        clause_id_counter = 1

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text()

            if not text:
                continue

            # Normalize text
            text = text.replace("\t", " ")
            text = re.sub(r"\s+", " ", text)

            # Split into pseudo-paragraphs
            paragraphs = re.split(r"\.\s+", text)

            for para in paragraphs:

                para = para.strip()

                if not self._is_valid_clause(para):
                    continue

                record_id = f"{file_name}_C{clause_id_counter}"

                properties = {
                    "source_document": file_name,
                    "page_number": page_number + 1,
                    "text": para
                }

                record = create_layer_record(
                    record_id=record_id,
                    entity_type="RegulationClause",
                    layer="L4",
                    category="General",
                    properties=properties
                )

                records.append(record)
                clause_id_counter += 1

        return records

    # --------------------------------------------------------
    # Noise Filtering Logic
    # --------------------------------------------------------

    def _is_valid_clause(self, text):

        if not text:
            return False

        # Remove extremely short lines
        if len(text) < 60:
            return False

        # Remove URLs
        if "www." in text.lower():
            return False

        # Remove copyright symbols
        if "©" in text or "copyright" in text.lower():
            return False

        # Remove lines that are mostly numbers
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.4:
            return False

        # Remove fully uppercase short lines (likely headers)
        if text.isupper() and len(text.split()) < 12:
            return False

        # Remove lines with too many special characters
        special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
        if special_ratio > 0.3:
            return False

        # Remove address-like lines
        if "marg" in text.lower() or "bhavan" in text.lower():
            return False

        return True