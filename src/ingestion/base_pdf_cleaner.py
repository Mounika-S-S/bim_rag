import pdfplumber
import re


class BasePDFCleaner:

    def extract_text(self, pdf_path):

        full_text = []

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                text = page.extract_text(x_tolerance=2, y_tolerance=2)

                if not text:
                    continue

                cleaned = self._clean_page(text)

                if cleaned:
                    full_text.append(cleaned)

        return "\n".join(full_text)

    def _clean_page(self, text):

        # remove page headers
        text = re.sub(
            r"TAMIL NADU GOVERNMENT GAZETTE.*",
            "",
            text,
            flags=re.IGNORECASE
        )

        # remove page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # remove table of contents lines
        text = re.sub(r"\.{3,}\s*\d+", "", text)

        # remove formulas
        text = re.sub(r"[A-Za-z]\s*=\s*\d+\s*N/mm.*", "", text)

        # remove repeated columns
        text = re.sub(r"\b\d{2,}\s+\d{2,}\b", "", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()