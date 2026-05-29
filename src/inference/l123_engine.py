# src/inference/l123_engine.py

class L123Engine:

    def __init__(self, l1_data, l2_data, l3_data):
        self.l1 = l1_data
        self.l2 = l2_data
        self.l3 = l3_data

    def run(self):

        results = []

        for element in self.l1:

            element_name = element.get("properties", {}).get("Name", "")

            for product in self.l2:

                product_name = product.get("properties", {}).get("Product_Name", "")

                if element_name and product_name and element_name.lower() in product_name.lower():

                    for process in self.l3:

                        process_text = process.get("properties", {}).get("text", "")

                        record = {
                            "element_id": element["id"],
                            "element_name": element_name,
                            "product": product_name,
                            "process_rule": process_text
                        }

                        results.append(record)

        return results