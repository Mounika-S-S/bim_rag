class ComplianceEngine:

    def __init__(self, l1_records, l2_records, l4_records):
        self.l1 = l1_records
        self.l2 = l2_records
        self.l4 = l4_records

    # --------------------------------------------------

    def run(self):

        results = []

        for element in self.l1:

            element_props = element.get("properties", {})
            element_id = element.get("id")
            element_name = element_props.get("Name", "")
            element_type = element.get("entity_type", "")

            product = self._match_product(element)

            if not product:
                continue

            product_props = product.get("properties", {})

            # ---------------------------
            # Check ALL L4 rules
            # ---------------------------

            for rule in self.l4:

                if not rule.get("is_numeric_rule"):
                    continue

                rule_text = rule.get("text", "")
                threshold = rule.get("threshold_value")
                operator = rule.get("comparison_operator")
                unit = rule.get("unit")

                if not rule_text or threshold is None or not operator:
                    continue

                # =========================
                # 🔥 SMART FIELD MAPPING
                # =========================
                field = self._map_rule_to_field(rule_text)

                if not field:
                    continue

                value = product_props.get(field)

                if value is None:
                    continue

                try:
                    value = float(value)
                except:
                    continue

                # =========================
                # COMPARISON
                # =========================
                violation = False

                if operator == ">=":
                    violation = value < threshold
                elif operator == "<=":
                    violation = value > threshold
                elif operator == ">":
                    violation = value <= threshold
                elif operator == "<":
                    violation = value >= threshold
                elif operator == "==":
                    violation = value != threshold

                # =========================
                # ALWAYS ADD RECORD
                # =========================
                results.append({
                    "element_id": element_id,
                    "element_name": element_name,
                    "element_type": element_type,

                    "is_compliant": not violation,

                    "rule_text": rule_text,
                    "product_value": value,
                    "required": f"{operator} {threshold}",
                    "unit": unit,
                    "field": field,

                    "threshold_value": threshold,
                    "comparison_operator": operator,
                    "layer_origin": "L4",
                })

        return results

    # --------------------------------------------------

    def _match_product(self, element):

        element_name = element.get("properties", {}).get("Name", "")

        for product in self.l2:
            product_name = product.get("properties", {}).get("Product_Name", "")
            if element_name and product_name and element_name in product_name:
                return product

        return None

    # --------------------------------------------------

    def _map_rule_to_field(self, rule_text):

        text = rule_text.lower()

        if "fire" in text:
            return "Fire_Rating_Hours"
        elif "height" in text:
            return "Clear_Height_m"
        elif "road" in text or "width" in text:
            return "Road_Width_m"
        elif "litre" in text or "oil" in text:
            return "Oil_Capacity"
        elif "distance" in text or "meter" in text:
            return "Distance_m"

        return None
