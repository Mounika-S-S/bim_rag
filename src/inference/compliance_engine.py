# src/inference/compliance_engine.py
"""
Deterministic Compliance Engine.
Checks L1 (IFC) + L2 (Product) elements against L4 (Regulations) and L5 (Requirements).

Strategy:
- L1 and L2 records are matched via element_type_normalized field
- Properties are merged (L2 fills gaps missing in L1)
- L4 rules matched by element_types list on each rule
- FireRating_min parsed from "120min", "120 min", 120 int/float
- Rich output: status, gap, suggestion, source_rule
"""
import re
from typing import List, Dict, Optional, Any

NUMBER_RE = re.compile(r"[\d]+(?:\.\d+)?")

# Canonical property → L4 rule type keywords that govern it
PROPERTY_RULE_MAP = {
    "FireRating_min": "FireRating",
    "Thickness_mm": ["Setback", "Height", "Structural"],
    "Height_mm": "Height",
    "Width_mm": ["Corridor", "Staircase"],
    "CompressiveStrength_MPa": "Structural",
    "Cover_mm": "Structural",
}

# Suggestions per rule type when non-compliant
SUGGESTIONS = {
    "FireRating": "Upgrade to a product with higher fire resistance rating meeting the required {req}{unit}.",
    "Height": "Review design height against the regulatory limit of {req}{unit}.",
    "Setback": "Adjust building setback to comply with minimum {req}{unit} requirement.",
    "FSI": "Recalculate floor space index to remain within the permitted {req} FSI.",
    "Parking": "Ensure parking provision meets the '1 per {req} {unit}' regulation.",
    "Corridor": "Widen corridor to minimum {req}{unit} as required.",
    "Staircase": "Adjust staircase width/riser to comply with {req}{unit} code requirement.",
    "Sanitation": "Add required sanitation facilities as per the {req} {unit} rule.",
    "Structural": "Consult structural engineer to meet minimum strength {req}{unit}.",
    "General": "Review the clause and update design accordingly.",
}


def _to_float(val: Any) -> Optional[float]:
    """Parse fire rating strings like '120min', '120 min', 2.0, '60', etc."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    m = NUMBER_RE.search(s)
    if m:
        return float(m.group())
    return None


def _compare(actual: float, threshold: float, operator: str) -> bool:
    """Return True if actual satisfies the rule (True = compliant)."""
    if operator == ">=":
        return actual >= threshold
    if operator == "<=":
        return actual <= threshold
    if operator == ">":
        return actual > threshold
    if operator == "<":
        return actual < threshold
    if operator == "==":
        return actual == threshold
    return True


def _suggestion(rule_type: str, req: float, unit: str) -> str:
    tpl = SUGGESTIONS.get(rule_type, SUGGESTIONS["General"])
    return tpl.format(req=req, unit=f" {unit}" if unit else "")


class ComplianceEngine:

    def __init__(self, l1_records: List[Dict], l2_records: List[Dict],
                 l4_records: List[Dict], l5_records: Optional[List[Dict]] = None):
        self.l1 = l1_records
        self.l2 = l2_records
        self.l4 = l4_records
        self.l5 = l5_records or []

        # Build lookup: element_type_normalized → [L2 records]
        self._l2_by_type: Dict[str, List[Dict]] = {}
        for rec in self.l2:
            etype = rec.get("element_type_normalized") or "unknown"
            self._l2_by_type.setdefault(etype, []).append(rec)

    # ------------------------------------------------------------------
    def run(self) -> List[Dict]:
        results = []
        for element in self.l1:
            results.extend(self._check_element(element))
        return results

    # ------------------------------------------------------------------
    def _check_element(self, element: Dict) -> List[Dict]:
        issues = []
        el_id = element.get("id", "")
        el_name = element.get("properties", {}).get("Name") or el_id
        el_type = element.get("entity_type", "")
        el_norm = element.get("element_type_normalized", el_type.lower())

        # Merge L1 + best-matching L2 properties
        merged_props = dict(element.get("properties", {}))
        l2_match = self._find_l2(el_norm, merged_props)
        if l2_match:
            for k, v in l2_match.get("properties", {}).items():
                if k not in merged_props or merged_props[k] is None:
                    merged_props[k] = v  # L2 fills gaps

        # Check against L4 regulations
        for rule in self.l4:
            if not rule.get("is_numeric_rule"):
                continue
            rule_etypes = rule.get("element_types", ["building"])
            # Match rule to element
            if el_norm not in rule_etypes and "building" not in rule_etypes:
                continue

            issue = self._evaluate_rule(rule, merged_props, el_id, el_name, el_type, "L4")
            if issue:
                issues.append(issue)

        # Check against L5 requirements
        for req in self.l5:
            props = req.get("properties", {})
            desc = str(props.get("description", ""))
            if not desc:
                continue
            # Simple text match — if requirement mentions element type
            if el_norm not in desc.lower() and el_type.lower() not in desc.lower():
                continue
            # check if any numeric threshold mentioned in description
            m_num = NUMBER_RE.search(desc)
            if m_num:
                threshold = float(m_num.group())
                # build a pseudo-rule
                pseudo_rule = {
                    "rule_type": "Requirement",
                    "is_numeric_rule": True,
                    "comparison_operator": ">=",
                    "threshold_value": threshold,
                    "unit": props.get("unit", ""),
                    "element_types": [el_norm],
                    "text": desc,
                }
                issue = self._evaluate_rule(pseudo_rule, merged_props, el_id, el_name, el_type, "L5")
                if issue:
                    issues.append(issue)

        # If no issues found, record compliant status for property summary
        if not issues:
            issues.append(self._compliant_record(el_id, el_name, el_type, merged_props))

        return issues

    # ------------------------------------------------------------------
    def _find_l2(self, el_norm: str, el_props: Dict) -> Optional[Dict]:
        """Find best-matching L2 product for this element type."""
        candidates = self._l2_by_type.get(el_norm, [])
        if candidates:
            return candidates[0]   # first match (can be improved with fuzzy)
        # Try partial match
        for etype, recs in self._l2_by_type.items():
            if etype in el_norm or el_norm in etype:
                return recs[0]
        return None

    # ------------------------------------------------------------------
    def _evaluate_rule(self, rule: Dict, props: Dict,
                       el_id: str, el_name: str, el_type: str,
                       source_layer: str) -> Optional[Dict]:
        """Evaluate one numeric rule against merged element properties. Returns issue dict or None."""
        rule_type = rule.get("rule_type", "General")
        threshold = rule.get("threshold_value")
        operator = rule.get("comparison_operator", ">=")
        unit = rule.get("unit", "")
        rule_text = rule.get("text", "")

        if threshold is None:
            return None

        # Determine which property to check for this rule type
        prop_key = self._rule_type_to_prop(rule_type, rule_text)
        if not prop_key:
            return None

        actual_raw = props.get(prop_key)
        actual = _to_float(actual_raw)

        if actual is None:
            return {
                "element_id": el_id,
                "element_name": el_name,
                "element_type": el_type,
                "property": prop_key,
                "l1_value": props.get(prop_key),
                "l2_value": None,
                "effective_value": None,
                "required_value": threshold,
                "operator": operator,
                "unit": unit,
                "source_rule": rule_text[:300],
                "source_layer": source_layer,
                "status": "MISSING_PROPERTY",
                "gap": None,
                "suggestion": f"Property '{prop_key}' not found in L1 or L2 data for this element. Add it to the product data.",
            }

        compliant = _compare(actual, threshold, operator)
        gap = round(actual - threshold, 3) if not compliant else None

        if compliant:
            return None   # No issue

        return {
            "element_id": el_id,
            "element_name": el_name,
            "element_type": el_type,
            "property": prop_key,
            "l1_value": props.get(prop_key),
            "l2_value": None,
            "effective_value": actual,
            "required_value": threshold,
            "operator": operator,
            "unit": unit,
            "source_rule": rule_text[:300],
            "source_layer": source_layer,
            "status": "NON_COMPLIANT",
            "gap": gap,
            "suggestion": _suggestion(rule_type, threshold, unit),
        }

    def _rule_type_to_prop(self, rule_type: str, text: str) -> Optional[str]:
        """Map rule_type + text keywords to the canonical property name to check."""
        t = text.lower()
        if rule_type == "FireRating" or "fire" in t:
            return "FireRating_min"
        if rule_type == "Height" or "height" in t:
            return "Height_mm"
        if rule_type == "Setback" or "setback" in t:
            return "Thickness_mm"
        if rule_type == "Corridor" or "corridor" in t or "width" in t:
            return "Width_mm"
        if rule_type == "Staircase" or "stair" in t:
            return "Width_mm"
        if rule_type == "Structural" or "compressive" in t or "strength" in t:
            return "CompressiveStrength_MPa"
        if rule_type == "Coverage" and "coverage" in t:
            return "Qty_Area"
        return None

    def _compliant_record(self, el_id: str, el_name: str, el_type: str, props: Dict) -> Dict:
        return {
            "element_id": el_id,
            "element_name": el_name,
            "element_type": el_type,
            "property": "ALL_CHECKED",
            "effective_value": None,
            "required_value": None,
            "operator": None,
            "unit": None,
            "source_rule": None,
            "source_layer": "L4+L5",
            "status": "COMPLIANT",
            "gap": None,
            "suggestion": None,
            "properties_summary": {k: v for k, v in props.items()
                                   if k not in ("Name", "ObjectType") and v is not None},
        }