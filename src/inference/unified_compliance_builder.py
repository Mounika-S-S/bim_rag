# src/inference/unified_compliance_builder.py

import uuid
from src.inference.compliance_engine import ComplianceEngine
from src.inference.l125_engine import L125Engine
from src.inference.l45_engine import L45Engine


class UnifiedComplianceBuilder:
    """
    Merges L124 + L125 + L45 inference into a single rich compliance.json.
    Each record explains:
      - compliance_status
      - which layers are involved
      - why the value is required (tracing to L4 code text or L5 company rule)
      - actual vs required value
    """

    def __init__(self, l1, l2, l3, l4, l5):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5

    def build(self):
        records = []
        counter = 1

        # ── L1-L2-L4 (Compliance Engine) ──────────────────────────────────
        engine_124 = ComplianceEngine(self.l1, self.l2, self.l4)
        raw_124 = engine_124.run()

        for r in raw_124:
            rule_text = r.get("rule_text", "")
            # Find source origin from L4 records for richer 'why_value_required'
            origin = self._find_l4_origin(rule_text)

            records.append({
                "compliance_id": f"CMP-{counter:04d}",
                "compliance_status": "NON_COMPLIANT",
                "element": {
                    "id": r.get("element_id"),
                    "name": r.get("element_name"),
                    "type": r.get("element_type"),
                    "layer": "L1"
                },
                "product": self._resolve_product(r.get("element_name")),
                "layers_involved": ["L1", "L2", "L4"],
                "why_non_compliant": (
                    f"Element '{r.get('element_name')}' has {r.get('unit', '')} value "
                    f"of {r.get('product_value')} but regulation requires {r.get('required')} {r.get('unit', '')}. "
                    f"Rule: {rule_text[:120]}..."
                ),
                "required_value": {
                    "operator": r.get("required", "").split(" ")[0] if r.get("required") else None,
                    "value": float(r.get("required", "0").split(" ")[-1]) if r.get("required") else None,
                    "unit": r.get("unit"),
                    "field": self._guess_field(rule_text)
                },
                "actual_value": {
                    "value": r.get("product_value"),
                    "unit": r.get("unit")
                },
                "why_value_required": (
                    f"L4 Code-book ({origin}) mandates: \"{rule_text[:300]}\" "
                    f"This threshold exists to ensure structural and fire safety of the building."
                ),
                "source_rule": {
                    "layer": "L4",
                    "rule_text": rule_text,
                    "origin": origin
                },
                "inference_type": "L1-L2-L4"
            })
            counter += 1

        # ── L1-L2-L5 (Company Requirements) ────────────────────────────────
        engine_125 = L125Engine(self.l1, self.l2, self.l5)
        raw_125 = engine_125.run()

        for r in raw_125:
            req_text = r.get("requirement", "")
            records.append({
                "compliance_id": f"CMP-{counter:04d}",
                "compliance_status": "REQUIREMENT_MAPPED",
                "element": {
                    "id": r.get("element_id"),
                    "name": r.get("element_name"),
                    "type": None,
                    "layer": "L1"
                },
                "product": {
                    "name": r.get("product"),
                    "layer": "L2"
                },
                "layers_involved": ["L1", "L2", "L5"],
                "why_non_compliant": (
                    f"Element '{r.get('element_name')}' using product '{r.get('product')}' "
                    f"is mapped to a company requirement that must be verified: {req_text[:120]}"
                ),
                "required_value": None,
                "actual_value": None,
                "why_value_required": (
                    f"L5 Company Rule: \"{req_text}\" "
                    f"This is a project/company-specific requirement added by the client or project manager "
                    f"(Layer 5) beyond what the code-books (Layer 4) mandate."
                ),
                "source_rule": {
                    "layer": "L5",
                    "rule_text": req_text,
                    "origin": "Company Requirements (L5)"
                },
                "inference_type": "L1-L2-L5"
            })
            counter += 1

        # ── L4-L5 (Regulation ↔ Company Rule Mapping) ──────────────────────
        engine_45 = L45Engine(self.l4, self.l5)
        raw_45 = engine_45.run()

        for r in raw_45:
            reg_text = r.get("regulation_clause", "")
            req_text = r.get("requirement", "")
            origin = self._find_l4_origin(reg_text)

            records.append({
                "compliance_id": f"CMP-{counter:04d}",
                "compliance_status": "REGULATION_REQUIREMENT_LINKED",
                "element": None,
                "product": None,
                "layers_involved": ["L4", "L5"],
                "why_non_compliant": (
                    f"A national regulation clause is linked to a company requirement. "
                    f"Both must be satisfied. Regulation: {reg_text[:120]}..."
                ),
                "required_value": None,
                "actual_value": None,
                "why_value_required": (
                    f"L4 Regulation ({origin}): \"{reg_text[:200]}\" "
                    f"This rule is operationalized by L5 Company Rule: \"{req_text[:200]}\" "
                    f"The L4 code-book sets the legal minimum; L5 adds project-specific enforcement."
                ),
                "source_rule": {
                    "layer": "L4",
                    "rule_text": reg_text,
                    "origin": origin
                },
                "linked_requirement": {
                    "layer": "L5",
                    "rule_text": req_text
                },
                "inference_type": "L4-L5"
            })
            counter += 1

        return records

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _resolve_product(self, element_name):
        if not element_name:
            return None
        for p in self.l2:
            pname = p.get("properties", {}).get("Product_Name", "")
            if element_name and pname and element_name in pname:
                return {"name": pname, "layer": "L2"}
        return {"name": "Unknown", "layer": "L2"}

    def _find_l4_origin(self, rule_text):
        rule_lower = (rule_text or "").lower()
        if "tangedco" in rule_lower:
            return "TANGEDCO Standards"
        if "fire and rescue" in rule_lower:
            return "Directorate of Fire and Rescue Services"
        if "national building code" in rule_lower or "nbc" in rule_lower:
            return "National Building Code of India 2016"
        if "tamil nadu" in rule_lower or "tncbr" in rule_lower:
            return "Tamil Nadu Combined Building Rules"
        if "is:" in rule_lower or "is 800" in rule_lower:
            return "Bureau of Indian Standards (BIS)"
        return "Building Regulation (L4)"

    def _guess_field(self, rule_text):
        rule_lower = (rule_text or "").lower()
        if "fire" in rule_lower:
            return "Fire_Rating_Hours"
        if "width" in rule_lower or "road" in rule_lower:
            return "Road_Width_m"
        if "height" in rule_lower or "basement" in rule_lower:
            return "Clear_Height_m"
        if "setback" in rule_lower:
            return "Setback_m"
        return "Measured_Value"
