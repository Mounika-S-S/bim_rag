import os
import json
import pandas as pd
import pdfplumber


class ProductExtractor:

    def __init__(self):
        pass

    # ============================================================
    # EXCEL PARSER
    # ============================================================

    def extract_from_excel(self, excel_path: str):

        df = pd.read_excel(excel_path)

        df.columns = [col.strip().lower() for col in df.columns]

        products = []

        for index, row in df.iterrows():

            product = {
                "id": f"L2_{index}",
                "entity_type": "Product",
                "layer": "L2",
                "category": self._safe_get(row, "category"),
                "properties": {
                    "ProductName": self._safe_get(row, "productname"),
                    "Material": self._safe_get(row, "material"),
                    "Height": self._to_float(self._safe_get(row, "height")),
                    "Length": self._to_float(self._safe_get(row, "length")),
                    "Width": self._to_float(self._safe_get(row, "width")),
                    "FireRating": self._safe_get(row, "firerating"),
                    "UValue": self._to_float(self._safe_get(row, "uvalue"))
                }
            }

            products.append(product)

        return products

    # ============================================================
    # PDF PARSER
    # ============================================================

    def extract_from_pdf(self, pdf_path: str):

        products = []

        with pdfplumber.open(pdf_path) as pdf:

            full_text = ""

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        # Very simple heuristic parsing
        # Assumes format like:
        # Product: XYZ
        # Height: 3.2 m
        # Material: Concrete

        entries = full_text.split("Product:")

        for idx, entry in enumerate(entries[1:]):

            product = {
                "id": f"L2_PDF_{idx}",
                "entity_type": "Product",
                "layer": "L2",
                "category": "Unknown",
                "properties": {}
            }

            lines = entry.split("\n")

            for line in lines:

                line_lower = line.lower()

                if "material" in line_lower:
                    product["properties"]["Material"] = self._extract_value(line)

                if "height" in line_lower:
                    product["properties"]["Height"] = self._extract_numeric(line)

                if "length" in line_lower:
                    product["properties"]["Length"] = self._extract_numeric(line)

                if "width" in line_lower:
                    product["properties"]["Width"] = self._extract_numeric(line)

                if "fire" in line_lower:
                    product["properties"]["FireRating"] = self._extract_value(line)

            products.append(product)

        return products

    # ============================================================
    # SAVE FUNCTION
    # ============================================================

    def save(self, products, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=4)

        print(f"L2 Product JSON saved at: {output_path}")

    # ============================================================
    # HELPERS
    # ============================================================

    def _safe_get(self, row, column):
        return row[column] if column in row and pd.notna(row[column]) else None

    def _to_float(self, value):
        try:
            return float(value)
        except:
            return None

    def _extract_numeric(self, line):
        import re
        match = re.search(r"\d+(\.\d+)?", line)
        if match:
            return float(match.group())
        return None

    def _extract_value(self, line):
        parts = line.split(":")
        if len(parts) > 1:
            return parts[1].strip()
        return None