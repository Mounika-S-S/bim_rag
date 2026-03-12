# src/l5/clause_segmenter.py
import re
from src.ingestion.base_pdf_cleaner import BasePDFCleaner


class ClauseSegmenter:
    """
    Extracts table rows from PDF schedules of rates
    Expected format: WR-CODE Description Unit Rate
    """
    
    def __init__(self):
        self.cleaner = BasePDFCleaner()
        
        # Pattern for schedule of rates tables
        self.table_pattern = re.compile(
            r"(WR-[A-Z]\d{4})\s+"  # Code: WR-A0123
            r"(.*?)\s+"             # Description
            r"(Kg|No|Sq\.?m\.?|Ltr|Mtr|Hour|RM|Cu\.?m\.?|Qtl\.?|Each|Set)\s+"  # Unit
            r"(\d+\.?\d*)"          # Rate
        )
    
    def segment(self, pdf_path):
        """
        Extract table rows from PDF
        Returns list of dicts with code, description, unit, rate
        """
        text = self.cleaner.extract_text(pdf_path)
        
        rows = []
        matches = self.table_pattern.findall(text)
        
        for match in matches:
            rows.append({
                "code": match[0],
                "description": match[1].strip(),
                "unit": match[2],
                "rate": float(match[3]),
                "page_number": 1  # You might want to track actual page numbers
            })
        
        # If no table matches, try line-by-line parsing
        if not rows:
            rows = self._fallback_parse(text)
        
        return rows
    
    def _fallback_parse(self, text):
        """Fallback parser for non-standard table formats"""
        rows = []
        lines = text.split('\n')
        
        for line in lines:
            # Look for patterns like "WR-A0123" anywhere in line
            if 'WR-' in line:
                parts = line.split()
                if len(parts) >= 4:
                    # Try to extract rate (last number)
                    rate = None
                    for part in reversed(parts):
                        try:
                            rate = float(part)
                            break
                        except:
                            continue
                    
                    if rate:
                        rows.append({
                            "code": parts[0] if parts[0].startswith('WR-') else 'WR-UNKNOWN',
                            "description": ' '.join(parts[1:-2]) if len(parts) > 3 else line,
                            "unit": parts[-2] if len(parts) > 2 else 'Nr',
                            "rate": rate,
                            "page_number": 1
                        })
        
        return rows