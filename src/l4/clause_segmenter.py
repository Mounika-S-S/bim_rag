"""
L4 Clause Segmenter — robust multi-pattern clause boundary detection.
Handles Tamil Nadu building rules codebook format.
"""
import re
from src.l4.pdf_cleaner import PDFCleaner

# Patterns that indicate a new clause begins
CLAUSE_START_PATTERNS = [
    re.compile(r"^\s*\d+\.\s+[A-Z]"),                   # "3. Definitions—"
    re.compile(r"^\s*\(\d+[A-Za-z]?\)\s+"),              # "(1)", "(1A)"
    re.compile(r"^\s*\d+\.\d+\s+[A-Z]"),                 # "3.2 Fire—"
    re.compile(r"^\s*[A-Z][A-Z\s]{4,}\s*[.—:]"),         # "FIRE SAFETY:"
    re.compile(r"^\s*Provided\s+that\b", re.IGNORECASE), # "Provided that..."
    re.compile(r"^\s*Explanation[.—:]", re.IGNORECASE),  # "Explanation.—"
]

# Section headings that are noise (TOC entries, part headers, schedule headers)
SKIP_PATTERNS = [
    re.compile(r"PART\s+[IVX]+\b", re.IGNORECASE),
    re.compile(r"SCHEDULE\s+[IVX]+\b", re.IGNORECASE),
    re.compile(r"ANNEXURE\s+[IVX\d]+", re.IGNORECASE),
    re.compile(r"^FORM\s+[A-Z\d]", re.IGNORECASE),
]

# Keywords that must be present for a clause to be relevant
RELEVANT_KEYWORDS = [
    "shall", "minimum", "maximum", "not less", "not exceed", "must",
    "required", "provided", "height", "setback", "fsi", "floor space",
    "parking", "stair", "basement", "corridor", "access", "water",
    "sanitary", "fire", "structural", "construction", "building",
    "development", "floor", "area", "width", "length", "storey",
    "coverage", "green", "solar", "rainwater", "lift", "elevator",
]

# Noise phrases that alone constitute short, useless clauses
NOISE_SNIPPETS = [
    "same meaning as defined",
    "have the same meaning",
    "shall have the meaning",
    "words and expressions",
]


def _is_clause_start(line: str) -> bool:
    return any(p.match(line) for p in CLAUSE_START_PATTERNS)


def _is_skip(line: str) -> bool:
    return any(p.search(line) for p in SKIP_PATTERNS)


def _is_relevant(text: str) -> bool:
    t = text.lower()
    if any(ns in t for ns in NOISE_SNIPPETS):
        return False
    return any(kw in t for kw in RELEVANT_KEYWORDS)


class ClauseSegmenter:

    def __init__(self):
        self.cleaner = PDFCleaner()

    def segment(self, pdf_path: str) -> list[str]:
        raw_text = self.cleaner.clean(pdf_path)
        return self._segment_text(raw_text)

    def _segment_text(self, raw_text: str) -> list[str]:
        lines = raw_text.split("\n")
        clauses = []
        current_clause_lines = []

        for line in lines:
            stripped = line.rstrip()

            if not stripped:
                # blank line — might be clause separator
                if current_clause_lines:
                    current_clause_lines.append("")
                continue

            if _is_skip(stripped):
                continue

            if _is_clause_start(stripped):
                # Save accumulated clause
                if current_clause_lines:
                    clause_text = " ".join(
                        l for l in current_clause_lines if l.strip()
                    ).strip()
                    if len(clause_text) > 40 and _is_relevant(clause_text):
                        clauses.append(clause_text)
                current_clause_lines = [stripped]
            else:
                current_clause_lines.append(stripped)

        # Flush last clause
        if current_clause_lines:
            clause_text = " ".join(
                l for l in current_clause_lines if l.strip()
            ).strip()
            if len(clause_text) > 40 and _is_relevant(clause_text):
                clauses.append(clause_text)

        return clauses