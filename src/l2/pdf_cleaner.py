import pdfplumber
import re


class PDFCleaner:

    def clean(self, pdf_path: str):

        text = ""

        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                t = page.extract_text()

                if t:
                    text += t + "\n"

        return self._post_process(text)

    def _post_process(self, text):

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"Page \d+", "", text)

        return text.strip()