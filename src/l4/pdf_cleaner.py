import pdfplumber
import re


class PDFCleaner:

    def clean(self, pdf_path: str) -> str:

        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        return self._post_process(full_text)

    # -----------------------------------------------------

    def _post_process(self, text: str) -> str:

        # remove repeated gazette headers
        text = re.sub(r"TAMIL NADU GOVERNMENT GAZETTE EXTRAORDINARY", "", text, flags=re.IGNORECASE)

        # remove page numbers
        text = re.sub(r"\n\d+\n", "\n", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # sentence breaks
        text = re.sub(r"\.\s+", ".\n", text)

        return text.strip()