import re
from src.ingestion.base_pdf_cleaner import BasePDFCleaner

class PDFCleaner(BasePDFCleaner):

    def clean(self, pdf_path):
        # BasePDFCleaner.extract_text handles the noise pages, extraction and basic cleanup
        text = self.extract_text(pdf_path)
        return self._post_process(text)

    def _post_process(self, text):
        text = re.sub(r"FOREWORD.*?(?=\n\d)", "", text, flags=re.I | re.S)
        text = re.sub(r"CONTENTS.*?(?=\n\d)", "", text, flags=re.I | re.S)
        text = re.sub(r"ACKNOWLEDGEMENTS.*", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        return text.strip()