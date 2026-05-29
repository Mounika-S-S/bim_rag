# src/app.py

import os
import json
from dotenv import load_dotenv

from src.ingestion.ifc_parser import IFCParser
from src.l2.main_l2_pipeline import L2Pipeline
from src.l3.main_l3_pipeline import L3Pipeline
from src.l5.main_l5_pipeline import L5Pipeline
from src.ingestion.main_l4_pipeline import L4Pipeline
from src.core.json_storage import JSONStorage
from src.inference.compliance_engine import ComplianceEngine
from src.rag.unified_vector_store import UnifiedVectorStore
from src.reasoning.llm_reasoner import LLMReasoner
from src.retrieval.query_router import QueryRouter
from src.retrieval.retriever import Retriever
from src.core.model_manager import model_manager
from src.inference.unified_compliance_builder import UnifiedComplianceBuilder

load_dotenv()
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

    # Preload ML models for better performance
    print("Loading ML models... (this may take a moment on first run)")
    try:
        model_manager.preload_models()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Model loading failed: {e}")
        print("The app will load models on-demand, but performance may be slower.")

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
        print("3. Add Process Data (L3)")
        print("4. Add Regulation (L4)")
        print("5. Add Requirement (L5)")
        print("6. View Stored JSON Files")
        print("7. Run Unified Compliance Inference (L124 + L125 + L45 → compliance.json)")
         
        
        print("8. Build Unified Vector Store")
        print("9. Query Semantic Knowledge")
        print("10. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            ingest_l1(project_id)

        elif choice == "2":
            ingest_l2(project_id)

        elif choice == "3":
            ingest_l3(project_id)      # FIXED

        elif choice == "4":
            ingest_l4(project_id)

        elif choice == "5":
            ingest_l5(project_id)

        elif choice == "6":
            view_json_files(project_id)

        elif choice == "7":
            run_unified_compliance(project_id)


        elif choice == "8":
            build_vector_store(project_id)

        elif choice == "9":
            query_vector_store(project_id)

        elif choice == "10":
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
# L2 Product Ingestion
# =====================================================

def ingest_l2(project_id):

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

            from src.ingestion.product_parser import ProductExtractor
            extractor = ProductExtractor()
            records = extractor.extract_from_excel(file_path)

        elif choice == "2":

            pipeline = L2Pipeline()
            records = pipeline.parse(file_path)

        else:
            print("Invalid choice.")
            return

        if not records:
            print("No products extracted.")
            return

        # 🔹 Save ONLY new records
        JSONStorage.save(project_id, "L2_product.json", records)

        total = len(JSONStorage.load(project_id, "L2_product.json"))

        print(f"L2 JSON saved. Total records now: {total}")

    except Exception as e:
        print(f"Error processing product file: {e}")
#=====================================================
# L3 Process Ingestion
#=====================================================

def ingest_l3(project_id):

    print("\nChoose Process Input Type:")
    print("1. Excel")
    print("2. PDF")

    choice = input("Select option: ").strip()

    file_path = input("Enter process file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:

        if choice == "1":

            from src.ingestion.l3_process_parser import L3ProcessParser
            parser = L3ProcessParser()
            records = parser.parse_excel(file_path)

        elif choice == "2":

            pipeline = L3Pipeline()
            records = pipeline.parse(file_path)

        else:
            print("Invalid choice.")
            return

        if not records:
            print("No process rules extracted.")
            return

        JSONStorage.save(project_id, "L3_process.json", records)

        total = len(JSONStorage.load(project_id, "L3_process.json"))

        print(f"L3 JSON saved. Total records now: {total}")

    except Exception as e:
        print(f"Error processing process file: {e}")
# =====================================================
# L4 Regulation Ingestion
# =====================================================

def ingest_l4(project_id):

    pipeline = L4Pipeline()
    new_records = []

    print("Enter PDF file paths (type 'done' to finish):")

    while True:

        file_path = input("PDF path: ").strip()

        if file_path.lower() == "done":
            break

        if not os.path.exists(file_path):
            print("File not found.")
            continue

        records = pipeline.parse(file_path)

        new_records.extend(records)

        print(f"Parsed {len(records)} structured rules from {os.path.basename(file_path)}")

    if not new_records:
        print("No regulations parsed.")
        return

    JSONStorage.save(project_id, "L4_regulation.json", new_records)

    total = len(JSONStorage.load(project_id, "L4_regulation.json"))

    print(f"L4 JSON saved. Total structured clauses: {total}")

# =====================================================
# L5 Requirement Ingestion
# =====================================================

def ingest_l5(project_id):

    print("\nChoose Requirement Input Type:")
    print("1. Excel")
    print("2. PDF")

    choice = input("Select option: ").strip()

    file_path = input("Enter requirement file path: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:

        if choice == "1":

            from src.ingestion.l5_requirement_parser import L5RequirementParser
            parser = L5RequirementParser()
            records = parser.parse_excel(file_path)

        elif choice == "2":

            pipeline = L5Pipeline()
            records = pipeline.parse(file_path)

        else:
            print("Invalid choice.")
            return

        if not records:
            print("No requirements extracted.")
            return

        JSONStorage.save(project_id, "L5_requirement.json", records)

        total = len(JSONStorage.load(project_id, "L5_requirement.json"))

        print(f"L5 JSON saved. Total records now: {total}")

    except Exception as e:
        print(f"Error processing requirement file: {e}")

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

#==========mismatch layer1,2,4=========
def run_unified_compliance(project_id):
    l1 = JSONStorage.load(project_id, "L1_ifc.json")
    l2 = JSONStorage.load(project_id, "L2_product.json")
    l3 = JSONStorage.load(project_id, "L3_process.json")
    l4 = JSONStorage.load(project_id, "L4_regulation.json")
    l5 = JSONStorage.load(project_id, "L5_requirement.json")

    if not l1 or not l2 or not l4:
        print("Missing required layers (L1, L2, L4 minimum).")
        return

    builder = UnifiedComplianceBuilder(l1, l2, l3, l4, l5)
    records = builder.build()

    # Save as single compliance.json (overwrite, not append)
    path = os.path.join("data/processed", project_id, "compliance.json")
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(records, f, indent=2)

    print(f"Unified compliance.json saved. Total records: {len(records)}")

#=========build vector store===========
def build_vector_store(project_id):

    store = UnifiedVectorStore()
    store.build_from_project(project_id)

    path = f"data/processed/{project_id}/unified.index"
    store.save(path)
#=======query vector store=============
def query_vector_store(project_id):

    # CLI path: ask user, then route to API helper
    question = input("Enter your question: ")
    answer = query_vector_store_api(project_id, question)
    print("\nLLM Reasoning Result:\n")
    print(answer)


def query_vector_store_api(project_id, question, chat_context=None):
    """
    Main query entry point.

    Routing logic (no deterministic_router / UnifiedSemanticRouter):
      1. If the question is a LISTING or COUNTING query about non-compliant
         elements → answer directly from compliance.json (100% recall, no FAISS
         hallucination).
      2. For all other queries → use QueryRouter (semantic, 6-target) + FAISS
         retrieval + LLM reasoning.
    """
    # ── Step 1: detect listing / counting intent ──────────────────────────
    q_lower = question.lower()
    is_list_query = any(kw in q_lower for kw in [
        "show all", "list all", "all non-compliant", "all elements",
        "which elements", "how many", "count", "give me all",
        "non compliant elements", "non-compliant elements",
        "show non-compliant", "list non-compliant",
    ])

    if is_list_query and any(kw in q_lower for kw in [
        "non-compliant", "non compliant", "violation", "failed", "fail",
        "compliant", "compliance",
    ]):
        return _answer_compliance_list_from_json(project_id, question)

    # ── Step 2: semantic RAG path ─────────────────────────────────────────
    router = QueryRouter()
    retriever = Retriever()

    path = f"data/processed/{project_id}/unified.index"
    if not os.path.exists(path):
        raise FileNotFoundError("Vector store not built yet. Run option 8 first.")

    retriever.load_unified_store(path)
    retriever.clear_chroma_store()

    routing = router.route_query(question, mode="faiss")

    results = retriever.retrieve(question, routing)

    # Merge past chat context with retrieved chunks
    context_parts = []
    if chat_context:
        context_parts.append(chat_context)

    # ── FIXED: max_chunks dynamic scaling ──
    # Compliance route gets up to 10 chunks; other routes get 5.
    route = routing.get("route", "")
    max_chunks = 10 if route == "Compliance" else 5
    context_parts.extend(results[:max_chunks])

    context = "\n".join(context_parts).strip()

    llm = LLMReasoner()
    answer = llm.reason(question, context)

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Direct compliance listing — reads compliance.json, NO FAISS, 100% recall
# ─────────────────────────────────────────────────────────────────────────────

def _answer_compliance_list_from_json(project_id, question):
    """
    Answers 'show all non-compliant elements' type questions by scanning
    compliance.json directly.  This guarantees every element is covered.
    """
    compliance_path = os.path.join("data/processed", project_id, "compliance.json")
    if not os.path.exists(compliance_path):
        return "Compliance data not found. Please run Unified Compliance Inference first (option 7)."

    with open(compliance_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    q_lower = question.lower()

    # ---- Count-only query ----
    if "how many" in q_lower or "count" in q_lower:
        non_compliant = [r for r in records
                         if not r.get("compliance", {}).get("is_compliant", True)]
        unique_elements = {r.get("element", {}).get("name", "") for r in non_compliant}
        unique_elements.discard("")
        return (
            f"There are **{len(unique_elements)} non-compliant elements** "
            f"with a total of **{len(non_compliant)} violations** across all rules.\n\n"
            f"Non-compliant elements:\n" +
            "\n".join(f"  - {e}" for e in sorted(unique_elements))
        )

    # ---- Full listing query (show all / list all) ----
    non_compliant = [r for r in records
                     if not r.get("compliance", {}).get("is_compliant", True)]

    if not non_compliant:
        return "No non-compliant elements found in the project."

    # Group by element name → collect worst violations per element
    grouped = {}
    for r in non_compliant:
        name = r.get("element", {}).get("name", "Unknown")
        if name not in grouped:
            grouped[name] = []
        grouped[name].append(r)

    lines = [f"**{len(grouped)} non-compliant elements found ({len(non_compliant)} total violations):**\n"]

    for idx, (elem_name, violations) in enumerate(sorted(grouped.items()), 1):
        # Representative violation (first one)
        rep   = violations[0]
        comp  = rep.get("compliance", {})
        elem  = rep.get("element", {})
        prod  = rep.get("product", {})
        src   = rep.get("source", {})

        act   = comp.get("actual_value") or {}
        req   = comp.get("required_value") or {}
        layer = comp.get("layer_responsible", "?")
        reason= comp.get("reason", "")
        why   = comp.get("why_required", "")

        lines.append(f"{idx}. **{elem_name}** ({elem.get('type','?')})")
        lines.append(f"   - Product: {prod.get('name','N/A')}")
        lines.append(f"   - Layer Responsible: {layer}")
        lines.append(f"   - Violations: {len(violations)}")
        lines.append(f"   - Example: Actual={act.get('value','?')} {act.get('unit','')} | "
                     f"Required={req.get('operator','')}{req.get('value','?')} {req.get('unit','')}")
        if reason:
            lines.append(f"   - Reason: {reason}")
        if why:
            lines.append(f"   - Why Required: {why[:200]}...")
        lines.append("")

    return "\n".join(lines)
#====smart query=============
def smart_query(project_id):

    query = input("Enter your question: ").lower()

    l2 = JSONStorage.load(project_id, "L2_product.json")
    mismatch = JSONStorage.load(project_id, "mismatch.json")

    # ---------------------------------------------------
    # 1️⃣ NON-COMPLIANCE EXPLANATION
    # ---------------------------------------------------
    if "non compliant" in query or "non-compliant" in query:

        print("\nCompliance Issues:\n")

        found = False

        for issue in mismatch:
            element_name = issue.get("element_name", "").lower()

            if element_name and element_name in query:
                print(issue)
                found = True

        if not found:
            print("No specific element matched. Showing top compliance issues:")
            for issue in mismatch[:5]:
                print(issue)

        return

    # ---------------------------------------------------
    # 2️⃣ LIST ALL NON-COMPLIANT ELEMENTS
    # ---------------------------------------------------
    if "which" in query or "list" in query:

        if "non compliant" in query:

            print("\nNon-Compliant Elements:\n")

            unique_elements = set()

            for issue in mismatch:
                name = issue.get("element_name")
                if name:
                    unique_elements.add(name)

            for e in unique_elements:
                print("-", e)

            print("\nTotal:", len(unique_elements))
            return

    # ---------------------------------------------------
    # 3️⃣ COST QUERY
    # ---------------------------------------------------
    if "cost" in query or "price" in query:

        for product in l2:
            name = product.get("properties", {}).get("Product_Name", "").lower()

            if name and name in query:
                cost = product["properties"].get("Unit_Cost_INR")
                print(f"\nCost of {name}: {cost} INR")
                return

    # ---------------------------------------------------
    # 4️⃣ DEFAULT → SEMANTIC SEARCH
    # ---------------------------------------------------
    store = UnifiedVectorStore()
    path = f"data/processed/{project_id}/unified.index"

    store.load(path)

    results = store.search(query, k=15)

    filtered = []

    for r in results:
        text = r.lower()

        if "non-compliant" in text or "non compliant" in text or "l124 inference" in text:
            filtered.append(r)

    if not filtered:
        filtered = results[:5]

    print("\nTop Results:\n")

    for r in results:
        print("-" * 80)
        print(r)


def classify_query(query):

    q = query.lower()

    if "non compliant" in q or "non-compliant" in q:
        if "which" in q or "list" in q:
            return "LIST_NON_COMPLIANT"
        return "COMPLIANCE_EXPLANATION"

    if "cost" in q or "price" in q:
        return "COST_QUERY"

    return "REGULATION_QUERY"
#============l1-l2-l3 inference===========
def run_l123(project_id):

    l1 = JSONStorage.load(project_id, "L1_ifc.json")
    l2 = JSONStorage.load(project_id, "L2_product.json")
    l3 = JSONStorage.load(project_id, "L3_process.json")

    if not l1 or not l2 or not l3:
        print("Missing layers for L1-L2-L3 inference.")
        return

    engine = L123Engine(l1, l2, l3)

    result = engine.run()

    JSONStorage.save(project_id, "l123_inference.json", result)

    print("L1-L2-L3 inference complete.")

 
# =====================================================

if __name__ == "__main__":
    main()
