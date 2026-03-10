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

        text = re.sub(r"FOREWORD.*?(?=\n\d)", "", text, flags=re.I | re.S)

        text = re.sub(r"CONTENTS.*?(?=\n\d)", "", text, flags=re.I | re.S)

        text = re.sub(r"ACKNOWLEDGEMENTS.*", "", text, flags=re.I)

        text = re.sub(r"\s+", " ", text)

        return text.strip()