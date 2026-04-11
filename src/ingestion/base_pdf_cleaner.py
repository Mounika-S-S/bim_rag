import pdfplumber
import re

class BasePDFCleaner:
    
    def __init__(self, crop_margin=0.08, drop_empty_pages=True, skip_toc=True):
        self.crop_margin = crop_margin  # Top and bottom crop percentage
        self.drop_empty_pages = drop_empty_pages
        self.skip_toc = skip_toc

    def clean(self, pdf_path: str) -> str:
        return self.extract_text(pdf_path)

    def extract_text(self, pdf_path: str) -> str:
        full_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                
                # Crop Headers and Footers
                width = page.width
                height = page.height
                
                bounding_box = (
                    0, 
                    float(height * self.crop_margin), 
                    width, 
                    float(height * (1 - self.crop_margin))
                )
                
                try:
                    cropped_page = page.within_bbox(bounding_box)
                    # Extract text using advanced layout features
                    text = cropped_page.extract_text(x_tolerance=2, y_tolerance=2)
                except ValueError:
                    # In case of bounding box issues, fallback to whole page
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                
                if not text:
                    continue

                if self._is_noise_page(text):
                    continue

                cleaned = self._clean_page_text(text)

                if cleaned:
                    full_text.append(cleaned)

        return "\n".join(full_text)

    def _is_noise_page(self, text: str) -> bool:
        """Heuristic to detect TOC, Index, Preface pages."""
        if not self.skip_toc:
            return False
            
        text_lower = text.lower()
        
        # Heuristic 1: Prevalent dot leaders "........"
        if len(re.findall(r"\.{4,}", text)) > 3:
            return True
            
        # Heuristic 2: Preface or Index at the very top (first 100 characters)
        start_snippet = text_lower[:100]
        if "preface" in start_snippet or "index" in start_snippet or "table of contents" in start_snippet or "contents" in start_snippet:
            # make sure it's an isolated title.
            # but sometimes "Contents" could be part of text "The contents of the container".
            # if it's the very first word, it's a good sign.
            if start_snippet.strip().startswith("preface") or start_snippet.strip().startswith("index") or start_snippet.strip().startswith("table of contents")  or start_snippet.strip().startswith("contents"):
                return True
            
        return False

    def _clean_page_text(self, text: str) -> str:
        
        # Remove any lingering gazette headers
        text = re.sub(
            r"TAMIL NADU GOVERNMENT GAZETTE.*",
            "",
            text,
            flags=re.IGNORECASE
        )
        
        # Remove page numbers that didn't get cropped
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"^Page \d+", "", text, flags=re.IGNORECASE|re.MULTILINE)
        
        # Remove formulas
        text = re.sub(r"[A-Za-z]\s*=\s*\d+\s*N/mm.*", "", text)
        
        # Remove repeated columns (e.g. repeated numbers)
        text = re.sub(r"\b\d{2,}\s+\d{2,}\b", "", text)

        # Remove foreword / acknowledgements lingering
        text = re.sub(r"ACKNOWLEDGEMENTS.*", "", text, flags=re.IGNORECASE)

        # Basic sentence normalization (add newline after period + space)
        text = re.sub(r"\.\s+", ".\n", text)
        
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        
        return text.strip()