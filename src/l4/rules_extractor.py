import re
from typing import List, Dict


class RulesExtractor:

    def __init__(self):

        # dimensional rules (meters, sqm, etc.)
        self.dimension_pattern = re.compile(
            r"(\d+(\.\d+)?)\s*(m|metre|meter|sqm|sq\.?m|mm|cm|feet|ft|%|kld|litres?|kl)",
            re.IGNORECASE
        )

        # ratio rules (One for every 50 persons)
        self.ratio_pattern = re.compile(
            r"one\s+(for|per)\s+(every\s+)?(\d+)",
            re.IGNORECASE
        )

        # count rules (One per floor)
        self.count_pattern = re.compile(
            r"one\s+per\s+(floor|room|unit)",
            re.IGNORECASE
        )

    # -------------------------------------

    def extract(self, clauses: List[str]) -> List[Dict]:

        structured_rules = []

        for clause in clauses:
            rule = self._extract_rule(clause)
            structured_rules.append(rule)

        return structured_rules

    # -------------------------------------

    def _extract_rule(self, text: str) -> Dict:

        text_clean = text.strip()
        text_lower = text_clean.lower()

        # 1️⃣ Ratio rules
        ratio_match = self.ratio_pattern.search(text_lower)
        if ratio_match:
            return {
                "rule_type": "Ratio",
                "is_numeric_rule": True,
                "comparison_operator": "ratio",
                "threshold_value": int(ratio_match.group(3)),
                "unit": "per unit",
                "text": text_clean
            }

        # 2️⃣ Count rules
        count_match = self.count_pattern.search(text_lower)
        if count_match:
            return {
                "rule_type": "Count",
                "is_numeric_rule": True,
                "comparison_operator": ">=",
                "threshold_value": 1,
                "unit": count_match.group(1),
                "text": text_clean
            }

        # 3️⃣ Dimensional rules
        dim_match = self.dimension_pattern.search(text_lower)
        if dim_match:
            return {
                "rule_type": self._classify_rule(text_lower),
                "is_numeric_rule": True,
                "comparison_operator": self._infer_operator(text_lower),
                "threshold_value": float(dim_match.group(1)),
                "unit": dim_match.group(3),
                "text": text_clean
            }

        # 4️⃣ fallback
        return {
            "rule_type": "General",
            "is_numeric_rule": False,
            "comparison_operator": None,
            "threshold_value": None,
            "unit": None,
            "text": text_clean
        }

    # -------------------------------------

    def _infer_operator(self, text: str):

        if "minimum" in text or "not less" in text:
            return ">="

        if "maximum" in text or "not exceed" in text:
            return "<="

        if "exceeding" in text or "more than" in text:
            return ">"

        if "less than" in text:
            return "<"

        return ">="

    # -------------------------------------

    def _classify_rule(self, text: str):

        if "setback" in text:
            return "Setback"

        if "height" in text:
            return "Height"

        if "fsi" in text:
            return "FSI"

        if "parking" in text:
            return "Parking"

        if "stair" in text:
            return "Staircase"

        if "basement" in text:
            return "Basement"

        if "corridor" in text:
            return "Corridor"

        if "water" in text or "sanitary" in text:
            return "Sanitation"

        return "Numeric"