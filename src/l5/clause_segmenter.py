import re
from src.ingestion.base_pdf_cleaner import BasePDFCleaner


class ClauseSegmenter:

    def __init__(self):
        self.cleaner = BasePDFCleaner()

    def segment(self, pdf_path):

        text = self.cleaner.extract_text(pdf_path)

        rows = []

        pattern = re.compile(
            r"(WR-[A-Z]\d{4})\s+(.*?)\s+(Kg|No|Sq\.m\.|Ltr|Mtr|Hour|RM|Cu\.m\.|Qtl\.)\s+(\d+\.\d+)"
        )

        matches = pattern.findall(text)

        for m in matches:

            rows.append({
                "code": m[0],
                "description": m[1],
                "unit": m[2],
                "rate": float(m[3])
            })

        return rows