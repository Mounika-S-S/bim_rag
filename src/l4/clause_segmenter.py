import re
from src.l4.pdf_cleaner import PDFCleaner


class ClauseSegmenter:

    def __init__(self):
        self.cleaner = PDFCleaner()

    # -------------------------------------

    def segment(self, pdf_path: str):

        raw_text = self.cleaner.clean(pdf_path)

        # Split by numbered clauses
        clauses = re.split(r"\s(?=\(?\d+\))", raw_text)

        filtered_clauses = []

        for clause in clauses:

            clause = clause.strip()
            
            if not clause:
                continue

            # Skip Annexures
            if self._is_annexure(clause):
                continue

            # Keep only development / control clauses
            if not self._is_relevant_clause(clause):
                continue

            filtered_clauses.append(clause)

        return filtered_clauses

    # -------------------------------------

    def _is_annexure(self, text: str) -> bool:
        return "annexure" in text.lower()

    # -------------------------------------

    def _is_relevant_clause(self, text: str) -> bool:

        text_lower = text.lower()

        # Skip definitions
        if text_lower.startswith("definition") or "means" in text_lower[:40]:
            return False

        # Skip transitional rules
        if "shall not apply" in text_lower and "construction in progress" in text_lower:
            return False

        # Skip annexure
        if "annexure" in text_lower:
            return False

        # Keep if numeric or measurable
        keywords = [
            "minimum",
            "maximum",
            "shall",
            "not less",
            "not exceed",
            "per",
            "for every",
            "height",
            "setback",
            "fsi",
            "parking",
            "floor",
            "water closet",
            "drinking water",
            "sanitary",
            "access",
            "corridor",
            "stair",
            "basement"
        ]

        return any(keyword in text_lower for keyword in keywords)