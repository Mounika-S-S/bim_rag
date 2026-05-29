import pdfplumber
import pandas as pd
import os

from src.core.schema import create_layer_record


class L5RequirementParser:

    # =================================
    # PDF Parser (Schedule of Rates)
    # =================================
    def parse_pdf(self, pdf_path):

        records = []

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                tables = page.extract_tables()

                if not tables:
                    continue

                for table in tables:

                    for row in table:

                        if not row:
                            continue

                        text = " | ".join([str(cell) for cell in row if cell])

                        record = create_layer_record(
                            f"L5_{len(records)}",
                            "Requirement",
                            "L5",
                            "ProjectRequirement",
                            {
                                "text": text,
                                "source_document": os.path.basename(pdf_path),
                                "page_number": page_number
                            }
                        )

                        records.append(record)

        return records

    # =================================
    # Excel Parser
    # =================================
    def parse_excel(self, excel_path):

        df = pd.read_excel(excel_path)

        records = []

        for idx, row in df.iterrows():

            row_text = " | ".join([str(v) for v in row.values if pd.notna(v)])

            record = create_layer_record(
                f"L5_{idx}",
                "Requirement",
                "L5",
                "ProjectRequirement",
                {
                    "text": row_text,
                    "source_document": os.path.basename(excel_path),
                    "row_number": idx
                }
            )

            records.append(record)

        return records