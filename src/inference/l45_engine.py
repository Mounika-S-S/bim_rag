class L45Engine:

    def __init__(self, l4_records, l5_records):

        self.l4 = l4_records
        self.l5 = l5_records

    # ---------------------------------------------

    def run(self):

        results = []
        seen = set()

        for rule in self.l4:

            rule_text = rule.get("text", "")

            if not rule_text:
                continue

            rule_lower = rule_text.lower()

            for req in self.l5:

                req_text = req.get("properties", {}).get("requirement_text", "")

                if not req_text:
                    continue

                req_lower = req_text.lower()

                # simple keyword overlap
                words = req_lower.split()

                for w in words:

                    if w in rule_lower and len(w) > 4:

                        key = (rule_text, req_text)

                        if key not in seen:

                            results.append({
                                "regulation_clause": rule_text,
                                "requirement": req_text
                            })

                            seen.add(key)

                        break

        return results