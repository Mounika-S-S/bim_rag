import pandas as pd
import json
from pathlib import Path


EXCEL_PATH = "data/excel/Layer2_Product_Data.xlsx"
OUTPUT_PATH = "data/output/products.json"


def parse_products(excel_path: str):
    """
    Reads product systems from Excel and converts them into
    normalized product JSON objects.
    """
    df = pd.read_excel(excel_path)

    products = []

    for idx, row in df.iterrows():
        product = {
            "product_id": f"SYSTEM_{idx + 1}",
            "system_type": row["SystemType"],
            "application": row["Application"],
            "fire_rating": row["FireRating"],
            "insulation": row["Insulation"],
            "board_type": row["BoardType"],
            "manufacturer": "SINIAT",
            "constraints": [row["Notes"]] if pd.notna(row["Notes"]) else [],
            "source": {
                "excel": "Layer2_Product_Data.xlsx",
                "pdf": "Layer2_Product_Data_EN.pdf"
            }
        }
        products.append(product)

    return products


def save_products(products, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(products, f, indent=2)


if __name__ == "__main__":
    products = parse_products(EXCEL_PATH)
    save_products(products, OUTPUT_PATH)
    print(f"PHASE 2A COMPLETE — {len(products)} products saved to {OUTPUT_PATH}")
