import re
from src.ingestion.base_pdf_cleaner import BasePDFCleaner

class PDFCleaner(BasePDFCleaner):

    def clean(self, pdf_path: str):
        # BasePDFCleaner.extract_text handles the noise pages, extraction and basic cleanup
        text = self.extract_text(pdf_path)
        return self._post_process(text)

    def _post_process(self, text):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"Page \d+", "", text)
        return text.strip()