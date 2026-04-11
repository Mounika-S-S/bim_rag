"""
L4 Rules Extractor — canonical numeric extraction + element type tagging.
Every rule gets an element_types list so the compliance engine can match
rules to IFC elements without fragile string matching.
"""
import re
from typing import List, Dict, Optional

# Element keywords → normalized element type
ELEMENT_KEYWORDS = {
    "wall": "wall", "walls": "wall",
    "beam": "beam",
    "column": "column",
    "slab": "slab",
    "roof": "roof",
    "stair": "stair", "staircase": "stair", "stair flight": "stair",
    "basement": "basement",
    "corridor": "corridor",
    "parking": "parking",
    "lift": "lift", "elevator": "lift",
    "ramp": "ramp",
    "door": "door",
    "window": "window",
    "foundation": "foundation", "footing": "foundation",
    "building": "building",  # generic / site-level rules
    "floor": "floor",
    "plot": "plot",
    "site": "site",
    "road": "road",
    "drain": "drain",
}

# Property type classification
RULE_TYPE_KEYWORDS = {
    "setback": "Setback",
    "fsi": "FSI",
    "floor space index": "FSI",
    "parking": "Parking",
    "height": "Height",
    "stair": "Staircase",
    "basement": "Basement",
    "corridor": "Corridor",
    "fire": "FireRating",
    "sanitar": "Sanitation",
    "water": "WaterSupply",
    "drainage": "Drainage",
    "green": "GreenBuilding",
    "solar": "GreenBuilding",
    "rainwater": "WaterHarvesting",
    "lift": "Lift",
    "structural": "Structural",
    "coverage": "Coverage",
    "plot": "PlotSize",
    "road": "RoadAccess",
}

DIMENSION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(m|metre|meter|sqm|sq\.?m|mm|cm|feet|ft|%|kld|litres?|kl|lph|min)\b",
    re.IGNORECASE,
)
RATIO_RE = re.compile(r"one\s+(?:for|per)\s+(?:every\s+)?(\d+)", re.IGNORECASE)
COUNT_RE = re.compile(r"one\s+per\s+(floor|room|unit|storey)", re.IGNORECASE)


def _extract_element_types(text: str) -> List[str]:
    t = text.lower()
    found = []
    for kw, norm in ELEMENT_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", t):
            if norm not in found:
                found.append(norm)
    return found or ["building"]  # default to building-level rule


def _classify_rule(text: str) -> str:
    t = text.lower()
    for kw, rtype in RULE_TYPE_KEYWORDS.items():
        if kw in t:
            return rtype
    return "General"


def _infer_operator(text: str) -> str:
    t = text.lower()
    if "minimum" in t or "not less than" in t or "at least" in t:
        return ">="
    if "maximum" in t or "not exceed" in t or "not more than" in t:
        return "<="
    if "exceeding" in t or "more than" in t:
        return ">"
    if "less than" in t:
        return "<"
    return ">="


class RulesExtractor:

    def extract(self, clauses: List[str]) -> List[Dict]:
        return [self._extract_rule(c) for c in clauses]

    def _extract_rule(self, text: str) -> Dict:
        text_clean = text.strip()
        text_lower = text_clean.lower()

        element_types = _extract_element_types(text_lower)
        rule_type = _classify_rule(text_lower)

        # 1. Ratio rule: "one for every 50 persons"
        ratio_match = RATIO_RE.search(text_lower)
        if ratio_match:
            return {
                "rule_type": rule_type or "Ratio",
                "is_numeric_rule": True,
                "comparison_operator": "ratio",
                "threshold_value": int(ratio_match.group(1)),
                "unit": "per unit",
                "element_types": element_types,
                "text": text_clean,
            }

        # 2. Count rule: "one per floor"
        count_match = COUNT_RE.search(text_lower)
        if count_match:
            return {
                "rule_type": rule_type or "Count",
                "is_numeric_rule": True,
                "comparison_operator": ">=",
                "threshold_value": 1,
                "unit": count_match.group(1),
                "element_types": element_types,
                "text": text_clean,
            }

        # 3. Dimensional rule
        dim_match = DIMENSION_RE.search(text_lower)
        if dim_match:
            return {
                "rule_type": rule_type,
                "is_numeric_rule": True,
                "comparison_operator": _infer_operator(text_lower),
                "threshold_value": float(dim_match.group(1)),
                "unit": dim_match.group(2).lower(),
                "element_types": element_types,
                "text": text_clean,
            }

        # 4. General (qualitative)
        return {
            "rule_type": rule_type,
            "is_numeric_rule": False,
            "comparison_operator": None,
            "threshold_value": None,
            "unit": None,
            "element_types": element_types,
            "text": text_clean,
        }