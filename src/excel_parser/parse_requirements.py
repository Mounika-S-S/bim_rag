import pandas as pd
import json
from pathlib import Path


EXCEL_PATH = "data/excel/Layer5_Project_Requirements.xlsx"
OUTPUT_PATH = "data/output/rules.json"


def parse_requirements(excel_path: str):
    """
    Converts project requirements from Excel into normalized rules.
    """
    df = pd.read_excel(excel_path)

    rules = []

    for _, row in df.iterrows():
        rule = {
            "requirement_id": row["RequirementID"],
            "element_type": row["ElementType"],
            "scope": row["ElementScope"],
            "property": row["Property"],
            "operator": row["Operator"],
            "value": row["RequiredValue"],
            "unit": row["Unit"],
            "priority": row["Priority"],
            "description": row["Description"]
        }
        rules.append(rule)

    return rules


def save_rules(rules, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rules, f, indent=2)


if __name__ == "__main__":
    rules = parse_requirements(EXCEL_PATH)
    save_rules(rules, OUTPUT_PATH)
    print(f"LAYER 5 COMPLETE — {len(rules)} project rules saved")
