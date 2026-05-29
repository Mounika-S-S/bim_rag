import re
from src.ingestion.base_pdf_cleaner import BasePDFCleaner

class PDFCleaner(BasePDFCleaner):
    
    def _clean_page_text(self, text: str) -> str:
        text = super()._clean_page_text(text)
        
        text = re.sub(r"Annexure.*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Sl\.?\s*No.*", "", text)
        text = re.sub(r"[A-Z]-\d{4}.*", "", text)
        
        # normalize whitespace again if regex resulted in gaps
        text = re.sub(r"[ \t]+", " ", text)
        
        return text.strip()