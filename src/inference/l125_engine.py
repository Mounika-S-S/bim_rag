class L125Engine:

    def __init__(self, l1_records, l2_records, l5_records):
        self.l1 = l1_records
        self.l2 = l2_records
        self.l5 = l5_records

    # ---------------------------------------------

    def run(self):

        results = []
        seen = set()

        for element in self.l1:

            element_name = element.get("properties", {}).get("Name", "")
            element_id = element.get("id")

            product = self._match_product(element_name)

            if not product:
                continue

            product_name = product.get("properties", {}).get("Product_Name", "")

            for req in self.l5:

                req_text = req.get("properties", {}).get("requirement_text", "")

                if not req_text:
                    continue

                key = (element_id, req_text)

                if key in seen:
                    continue

                seen.add(key)

                results.append({
                    "element_id": element_id,
                    "element_name": element_name,
                    "product": product_name,
                    "requirement": req_text
                })

        return results

    # ---------------------------------------------

    def _match_product(self, element_name):

        for product in self.l2:

            pname = product.get("properties", {}).get("Product_Name", "")

            if element_name and pname and element_name in pname:
                return product

        return None