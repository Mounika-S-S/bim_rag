# src/app.py

import os
import json

from src.ingestion.ifc_parser import IFCParser
from src.ingestion.main_l4_pipeline import L4Pipeline
from src.ingestion.product_parser import ProductExtractor
from src.core.json_storage import JSONStorage


PROJECTS_PATH = "data/processed"


# =====================================================
# Project Handling
# =====================================================

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


# =====================================================
# Interactive Menu
# =====================================================

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


# =====================================================
# L1 IFC Ingestion
# =====================================================

def ingest_l1(project_id):

    file_path = input("Enter IFC file path (.ifc): ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    parser = IFCParser()
    records = parser.parse_ifc(file_path)

    JSONStorage.save(project_id, "L1_ifc.json", records)

    print(f"L1 JSON saved. Total records: {len(records)}")


# =====================================================
# L2 Product Ingestion (FIXED PROPERLY)
# =====================================================

def ingest_l2(project_id):

    extractor = ProductExtractor()

    print("\nChoose Product Input Type:")
    print("1. Excel")
    print("2. PDF")

    choice = input("Select option: ").strip()

    file_path = input("Enter product file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:
        if choice == "1":
            products = extractor.extract_from_excel(file_path)

        elif choice == "2":
            products = extractor.extract_from_pdf(file_path)

        else:
            print("Invalid choice.")
            return

        if not products:
            print("No products extracted.")
            return

        JSONStorage.save(project_id, "L2_product.json", products)

        print(f"L2 JSON saved. Total records: {len(products)}")

    except Exception as e:
        print(f"Error processing product file: {e}")


# =====================================================
# L4 Regulation Ingestion
# =====================================================

def ingest_l4(project_id):

    pipeline = L4Pipeline()

    all_records = []

    print("Enter PDF file paths (type 'done' to finish):")

    while True:

        file_path = input("PDF path: ").strip()

        if file_path.lower() == "done":
            break

        if not os.path.exists(file_path):
            print("File not found.")
            continue

        records = pipeline.parse(file_path)
        all_records.extend(records)

        print(f"Parsed {len(records)} structured rules from {os.path.basename(file_path)}")

    if not all_records:
        print("No regulations parsed.")
        return

    existing = JSONStorage.load(project_id, "L4_regulation.json")

    if existing:
        all_records = existing + all_records

    JSONStorage.save(project_id, "L4_regulation.json", all_records)

    print(f"L4 JSON saved. Total structured clauses: {len(all_records)}")


# =====================================================
# L5 Requirement Ingestion
# =====================================================

def ingest_l5(project_id):

    file_path = input("Enter Requirement JSON file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    JSONStorage.save(project_id, "L5_requirement.json", data)

    print(f"L5 JSON saved. Total records: {len(data)}")


# =====================================================
# View Stored JSON Files
# =====================================================

def view_json_files(project_id):

    project_path = os.path.join(PROJECTS_PATH, project_id)

    if not os.path.exists(project_path):
        print("Project folder not found.")
        return

    files = os.listdir(project_path)

    if not files:
        print("No JSON files stored yet.")
        return

    print("\nStored JSON Files:")
    for f in files:
        print(f" - {f}")


# =====================================================

if __name__ == "__main__":
    main()