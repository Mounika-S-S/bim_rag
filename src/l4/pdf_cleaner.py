import re
from src.ingestion.base_pdf_cleaner import BasePDFCleaner
import pdfplumber


class PDFCleaner(BasePDFCleaner):
    """
    L4-specific cleaner inheriting robust BasePDFCleaner.
    Returns full page text preserving clause structure (no sentence splitting).
    """

    def clean(self, pdf_path: str) -> str:
        """Extract and clean text, preserving multi-sentence clause structure."""
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                h, w = page.height, page.width
                try:
                    cropped = page.within_bbox((0, 0.07 * h, w, 0.93 * h))
                    text = cropped.extract_text(x_tolerance=2, y_tolerance=2)
                except Exception:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)

                if not text:
                    continue
                if self._is_noise_page(text):
                    continue

                text = self._clean_page(text)
                if text:
                    full_text.append(text)

        return "\n".join(full_text)

    def _clean_page(self, text: str) -> str:
        # Remove standalone page numbers only
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Remove known gazette header fragments that survive crop
        text = re.sub(r"TAMIL NADU GOVERNMENT GAZETTE\s*EXTRAORDINARY", "", text, flags=re.IGNORECASE)
        # Remove table-of-content dot leaders (e.g. "Clause 3 ...... 12")
        text = re.sub(r"\.{3,}\s*\d+", "", text)
        # Collapse excess blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()