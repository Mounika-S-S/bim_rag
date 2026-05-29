import uuid
from src.inference.compliance_engine import ComplianceEngine
from src.inference.l125_engine import L125Engine
from src.inference.l45_engine import L45Engine


class UnifiedComplianceBuilder:

    def __init__(self, l1, l2, l3, l4, l5):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5

    def build(self):

        records = []
        counter = 1

        # =========================
        # STEP 1: L124 (REAL COMPLIANCE)
        # =========================
        engine_124 = ComplianceEngine(self.l1, self.l2, self.l4)
        raw_124 = engine_124.run()

        # =========================
        # STEP 2: L125 (COMPANY RULES)
        # =========================
        engine_125 = L125Engine(self.l1, self.l2, self.l5)
        raw_125 = engine_125.run()

        # Map L125 by element
        l125_map = {}
        for r in raw_125:
            eid = r.get("element_id")
            l125_map.setdefault(eid, []).append(r.get("requirement"))

        # =========================
        # STEP 3: L45 (REG ↔ COMPANY LINK)
        # =========================
        engine_45 = L45Engine(self.l4, self.l5)
        raw_45 = engine_45.run()

        # Map L45 rules
        l45_map = {}
        for r in raw_45:
            reg = r.get("regulation_clause", "")
            req = r.get("requirement", "")
            if reg:
                l45_map.setdefault(reg, []).append(req)

        # =========================
        # STEP 4: MERGE PER ELEMENT
        # =========================
        for r in raw_124:

            element_id = r.get("element_id")
            element_name = r.get("element_name")
            rule_text = r.get("rule_text", "")
            origin = self._find_l4_origin(rule_text)

            # ---- Compliance Explanation ----
            if r.get("is_compliant"):
                explanation = (
                    f"Element satisfies requirement {r.get('required')} {r.get('unit')}"
                )
            else:
                explanation = (
                    f"Element has {r.get('product_value')} "
                    f"but requires {r.get('required')} {r.get('unit')}"
                )

            # ---- Required Parsing ----
            req = r.get("required", "")
            parts = req.split(" ") if req else []

            # ---- L5 rules (element-based) ----
            l5_rules = l125_map.get(element_id, [])

            # ---- L45 linked rules ----
            linked_l5 = l45_map.get(rule_text, [])

            # ---- WHY VALUE REQUIRED ----
            why_required = (
                f"As per L4 ({origin}): '{rule_text[:200]}'. "
                f"This ensures safety and compliance."
            )

            if linked_l5:
                why_required += (
                    f" This regulation is linked to L5 requirement: '{linked_l5[0][:120]}'."
                )

            if l5_rules:
                why_required += (
                    f" Additionally, project/company rule requires: '{l5_rules[0][:120]}'."
                )

            # ---- FINAL RECORD ----
            records.append({
                "compliance_id": f"CMP-{counter:04d}",

                "element": {
                    "id": element_id,
                    "name": element_name,
                    "type": r.get("element_type"),
                },

                "product": self._resolve_product(element_name),

                "compliance": {
                    "status": "COMPLIANT" if r.get("is_compliant") else "NON_COMPLIANT",
                    "is_compliant": r.get("is_compliant"),

                    # 🔥 FIXED LAYER
                    "layer_responsible": "L5" if l5_rules else "L4",

                    "actual_value": {
                        "value": r.get("product_value"),
                        "unit": r.get("unit"),
                        "field": r.get("field"),
                    },

                    "required_value": {
                        "operator": parts[0] if len(parts) > 0 else None,
                        "value": float(parts[-1]) if len(parts) > 1 else None,
                        "unit": r.get("unit"),
                        "field": r.get("field"),
                    },

                    "reason": explanation,

                    "why_required": why_required,

                    "l5_rules": l5_rules if l5_rules else None
                },

                "layers_involved": ["L1", "L2", "L4"] + (["L5"] if l5_rules else []),

                "source": {
                    "rule_text": rule_text,
                    "origin": origin
                }
            })

            counter += 1

        return records

    # =========================
    # HELPERS
    # =========================

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
            return "Fire and Rescue Services"
        if "national building code" in rule_lower:
            return "NBC 2016"
        if "tamil nadu" in rule_lower:
            return "Tamil Nadu Building Rules"
        if "is:" in rule_lower:
            return "BIS Code"

        return "Building Regulation"
