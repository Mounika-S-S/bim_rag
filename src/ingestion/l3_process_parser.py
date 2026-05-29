# src/ingestion/l3_process_parser.py

import pdfplumber
import pandas as pd
import re
import os
from src.core.schema import create_layer_record


class L3ProcessParser:

    def __init__(self):
        self.clause_pattern = re.compile(r'^\d+(\.\d+)*')

    # ------------------------------------------------
    # Clean BIS / header noise
    # ------------------------------------------------
    def clean_text(self, text):

        noise_words = [
            "BUREAU OF INDIAN STANDARDS",
            "Free Standard provided by BIS",
            "MANAK BHAVAN",
            "Price Group",
            "ICS",
            "FOREWORD"
        ]

        for n in noise_words:
            text = text.replace(n, "")

        return text.strip()

    # ------------------------------------------------
    # PDF Parser
    # ------------------------------------------------
    def parse_pdf(self, pdf_path):

        records = []
        clause = None
        clause_text = ""

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                text = page.extract_text()

                if not text:
                    continue

                text = self.clean_text(text)

                lines = text.split("\n")

                for line in lines:

                    line = line.strip()

                    if not line:
                        continue

                    # Detect clause number
                    if self.clause_pattern.match(line):

                        # Save previous clause
                        if clause and clause_text.strip():

                            records.append(
                                create_layer_record(
                                    record_id=f"L3_{len(records)}",
                                    entity_type="ProcessRule",
                                    layer="L3",
                                    category="ConstructionProcess",
                                    properties={
                                        "clause": clause,
                                        "text": clause_text.strip(),
                                        "source_document": os.path.basename(pdf_path),
                                        "page_number": page_number
                                    }
                                )
                            )

                        parts = line.split(" ", 1)

                        clause = parts[0]
                        clause_text = parts[1] if len(parts) > 1 else ""

                    else:
                        clause_text += " " + line

        # Save last clause
        if clause and clause_text.strip():

            records.append(
                create_layer_record(
                    record_id=f"L3_{len(records)}",
                    entity_type="ProcessRule",
                    layer="L3",
                    category="ConstructionProcess",
                    properties={
                        "clause": clause,
                        "text": clause_text.strip(),
                        "source_document": os.path.basename(pdf_path),
                        "page_number": page_number
                    }
                )
            )

        return records

    # ------------------------------------------------
    # Excel Parser
    # ------------------------------------------------
    def parse_excel(self, excel_path):

        df = pd.read_excel(excel_path)

        records = []

        for idx, row in df.iterrows():

            text = " | ".join([str(v) for v in row.values if pd.notna(v)])

            records.append(
                create_layer_record(
                    record_id=f"L3_{idx}",
                    entity_type="ProcessRule",
                    layer="L3",
                    category="ConstructionProcess",
                    properties={
                        "text": text,
                        "source_document": os.path.basename(excel_path),
                        "row_number": idx
                    }
                )
            )

        return records