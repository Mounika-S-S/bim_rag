import pdfplumber
import re


class PDFCleaner:

    def clean(self, pdf_path):

        text = ""

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                t = page.extract_text()

                if t:
                    text += t + "\n"

        return self._post_process(text)

    def _post_process(self, text):

        text = re.sub(r"Annexure.*", "", text, flags=re.I)

        text = re.sub(r"Sl\.?\s*No.*", "", text)

        text = re.sub(r"[A-Z]-\d{4}.*", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()