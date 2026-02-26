# src/app.py

import os
from src.ingestion.ifc_parser import IFCParser
from src.core.json_storage import JSONStorage
from src.ingestion.regulation_parser import RegulationParser

PROJECTS_PATH = "data/processed"


def list_projects():
    if not os.path.exists(PROJECTS_PATH):
        os.makedirs(PROJECTS_PATH, exist_ok=True)
        return []

    return [
        name for name in os.listdir(PROJECTS_PATH)
        if os.path.isdir(os.path.join(PROJECTS_PATH, name))
    ]


def main():

    print("\n=== BIM Compliance System ===\n")

    projects = list_projects()

    if projects:
        print("Existing Projects:")
        for p in projects:
            print(f" - {p}")
    else:
        print("No existing projects found.")

    project_name = input("\nEnter project name: ").strip()

    if project_name not in projects:
        print(f"\nCreating new project: {project_name}")
        os.makedirs(os.path.join(PROJECTS_PATH, project_name), exist_ok=True)
    else:
        print(f"\nOpening existing project: {project_name}")

    interactive_menu(project_name)


def interactive_menu(project_id):

    while True:

        print("\n====== MENU ======")
        print("1. Add IFC Model (L1)")
        print("2. Add Product Data (L2)")
        print("3. Add Regulation (L4)")
        print("4. Add Requirement (L5)")
        print("5. View Stored JSON Files")
        print("6. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            ingest_l1(project_id)

        elif choice == "2":
            ingest_l2(project_id)

        elif choice == "3":
            ingest_l4(project_id)

        elif choice == "4":
            ingest_l5(project_id)

        elif choice == "5":
            view_json_files(project_id)

        elif choice == "6":
            print("Exiting system.")
            break

        else:
            print("Invalid choice.")


# -------------------------
# L1 IFC Ingestion
# -------------------------

def ingest_l1(project_id):

    file_path = input("Enter IFC file path (.ifc): ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    parser = IFCParser()
    records = parser.parse_ifc(file_path)

    JSONStorage.save(project_id, "L1_ifc.json", records)

    print(f"L1 JSON saved. Total records: {len(records)}")


# -------------------------
# L2 Product Ingestion
# -------------------------

def ingest_l2(project_id):

    file_path = input("Enter Product JSON file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    import json
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    JSONStorage.save(project_id, "L2_product.json", data)

    print(f"L2 JSON saved. Total records: {len(data)}")


# -------------------------
# L4 Regulation Ingestion
# -------------------------

def ingest_l4(project_id):

    parser = RegulationParser()

    all_records = []

    print("Enter PDF file paths (type 'done' to finish):")

    while True:

        file_path = input("PDF path: ").strip()

        if file_path.lower() == "done":
            break

        if not os.path.exists(file_path):
            print("File not found.")
            continue

        records = parser.parse_pdf(file_path)
        all_records.extend(records)

        print(f"Parsed {len(records)} clauses from {os.path.basename(file_path)}")

    if not all_records:
        print("No regulations parsed.")
        return

    # Load existing if exists (append mode)
    existing = JSONStorage.load(project_id, "L4_regulation.json")

    if existing:
        all_records = existing + all_records

    JSONStorage.save(project_id, "L4_regulation.json", all_records)

    print(f"L4 JSON saved. Total clauses: {len(all_records)}")


# -------------------------
# L5 Requirement Ingestion
# -------------------------

def ingest_l5(project_id):

    file_path = input("Enter Requirement JSON file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    import json
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    JSONStorage.save(project_id, "L5_requirement.json", data)

    print(f"L5 JSON saved. Total records: {len(data)}")


# -------------------------
# View Stored JSON
# -------------------------

def view_json_files(project_id):

    project_path = os.path.join(PROJECTS_PATH, project_id)

    files = os.listdir(project_path)

    if not files:
        print("No JSON files stored yet.")
        return

    print("\nStored JSON Files:")
    for f in files:
        print(f" - {f}")


if __name__ == "__main__":
    main()