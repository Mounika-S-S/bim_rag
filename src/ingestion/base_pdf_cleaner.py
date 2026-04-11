import pdfplumber
import re


class BasePDFCleaner:

    def __init__(self):
        self.toc_patterns = [
            re.compile(r'\.{5,}\s*(?:[A-Za-z0-9]+)?'),  # Dot leaders with page number
            re.compile(r'(?i)^\s*(table of contents|contents|index)\s*$') # Explicit TOC headers
        ]
        self.preface_patterns = [
            re.compile(r'(?i)^\s*(preface|foreword|acknowledgements|introduction)\s*$')
        ]
        self.roman_numeral_pattern = re.compile(r'^\s*([ivxlcdm]+)\s*$', re.IGNORECASE)

    def extract_text(self, pdf_path):
        full_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                
                # Crop headers and footers (top 7% and bottom 7% dynamically)
                width = page.width
                height = page.height
                box = (0, 0.07 * height, width, 0.93 * height)
                
                try:
                    cropped_page = page.within_bbox(box)
                    text = cropped_page.extract_text(x_tolerance=2, y_tolerance=2)
                except Exception:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)

                if not text:
                    continue

                if self._is_noise_page(text):
                    continue

                cleaned = self._clean_page(text)

                if cleaned:
                    full_text.append(cleaned)

        return "\n".join(full_text)

    def _is_noise_page(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return True
            
        # 1. Romans page number only check
        if len(lines) <= 2 and all(self.roman_numeral_pattern.match(l) for l in lines):
            return True

        # 2. Check for Table of Contents lines
        toc_lines = sum(1 for line in lines if self.toc_patterns[0].search(line))
        if len(lines) > 5 and (toc_lines / len(lines)) > 0.1:
            return True
            
        # 3. Check for specific headings in the first few lines
        head = " \n ".join(lines[:5])
        if any(p.search(head) for p in self.toc_patterns[1:]):
            return True
        if any(p.search(head) for p in self.preface_patterns):
            return True
            
        return False

    def _clean_page(self, text):
        # We rely mostly on bbox cropping to remove headers/footers now
        
        # remove standalone page numbers that somehow snuck through the crop
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # remove formulas formatting that usually pollutes RAG
        text = re.sub(r"[A-Za-z]\s*=\s*\d+\s*N/mm.*", "", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()