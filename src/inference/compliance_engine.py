# src/inference/compliance_engine.py

class ComplianceEngine:

    def __init__(self, l1_records, l2_records, l4_records):
        self.l1 = l1_records
        self.l2 = l2_records
        self.l4 = l4_records

    # --------------------------------------------------

    def run(self):

        l124_inference = []

        for element in self.l1:

            element_props = element.get("properties", {})
            element_id = element.get("id")
            element_name = element_props.get("Name", "")
            element_type = element.get("entity_type", "")

            # ---------------------------
            # Match product
            # ---------------------------

            product = self._match_product(element)

            if not product:

                l124_inference.append({
                    "element_id": element_id,
                    "element_name": element_name,
                    "element_type": element_type,
                    "issue": "No matching product found",
                    "layer": "L2"
                })

                continue

            # ---------------------------
            # Check regulation compliance
            # ---------------------------

            reg_issues = self._check_regulation(product, element)

            l124_inference.extend(reg_issues)

        return l124_inference

    # --------------------------------------------------

    def _match_product(self, element):

        element_name = element.get("properties", {}).get("Name", "")

        for product in self.l2:

            product_props = product.get("properties", {})
            product_name = product_props.get("Product_Name", "")

            if element_name and product_name and element_name in product_name:
                return product

        return None

    # --------------------------------------------------

    def _check_regulation(self, product, element):

        issues = []

        product_props = product.get("properties", {})
        element_id = element.get("id")
        element_name = element.get("properties", {}).get("Name", "")
        element_type = element.get("entity_type", "")

        for rule in self.l4:

            if not rule.get("is_numeric_rule"):
                continue

            rule_text = rule.get("text")
            threshold = rule.get("threshold_value")
            operator = rule.get("comparison_operator")
            unit = rule.get("unit")

            if not rule_text or threshold is None or not operator:
                continue

            # Example: fire rule
            if "fire" in rule_text.lower():

                fire_rating = product_props.get("Fire_Rating_Hours")

                if fire_rating is None:
                    continue

                try:
                    fire_rating = float(fire_rating)
                except:
                    continue

                violation = False

                if operator == ">=" and fire_rating < threshold:
                    violation = True
                elif operator == "<=" and fire_rating > threshold:
                    violation = True
                elif operator == ">" and fire_rating <= threshold:
                    violation = True
                elif operator == "<" and fire_rating >= threshold:
                    violation = True
                elif operator == "==" and fire_rating != threshold:
                    violation = True

                if violation:

                    issues.append({
                        "element_id": element_id,
                        "element_name": element_name,
                        "element_type": element_type,
                        "rule_text": rule_text,
                        "product_value": fire_rating,
                        "required": f"{operator} {threshold}",
                        "unit": unit,
                        "issue": "Value does not meet regulatory requirement",
                        # ── NEW FIELDS ──
                        "layer_origin": "L4",
                        "comparison_operator": operator,
                        "threshold_value": threshold,
                    })

        return issues
