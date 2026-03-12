# src/app.py
"""
Main application - just menu and user interaction
All logic is in separate modules
"""
import os
from src.core.json_storage import JSONStorage
from src.ingestion.ifc_parser import IFCParser
from src.ingestion.product_parser import ProductExtractor  # Enhanced version
from src.ingestion.l5_requirement_parser import RequirementParser
from src.ingestion.main_l4_pipeline import L4Pipeline
from src.inference.compliance_engine import ComplianceEngine
from src.inference.compliance_engine_l5 import ComplianceEngineL5
from src.inference.compliance_engine_l4_l5 import ComplianceEngineL4L5
from src.rag.unified_vector_store import UnifiedVectorStore
from src.retrieval.query_router import QueryRouter
from datetime import datetime

PROJECTS_PATH = "data/processed"


# ==================== Project Management ====================

def list_projects():
    """List all existing projects"""
    if not os.path.exists(PROJECTS_PATH):
        os.makedirs(PROJECTS_PATH, exist_ok=True)
        return []
    return [name for name in os.listdir(PROJECTS_PATH) 
            if os.path.isdir(os.path.join(PROJECTS_PATH, name))]


def select_project():
    """Let user select or create a project"""
    projects = list_projects()
    
    print("\n=== BIM Compliance System ===\n")
    
    if projects:
        print("Existing Projects:")
        for p in projects:
            print(f"  - {p}")
    else:
        print("No existing projects found.")
    
    project_name = input("\nEnter project name: ").strip()
    
    if project_name not in projects:
        print(f"\n📁 Creating new project: {project_name}")
        os.makedirs(os.path.join(PROJECTS_PATH, project_name), exist_ok=True)
    else:
        print(f"\n📁 Opening project: {project_name}")
    
    return project_name


# ==================== Ingestion Functions ====================

def ingest_l1(project_id):
    """Add IFC model (L1)"""
    print("\n" + "="*60)
    print("📦 L1 IFC MODEL INGESTION")
    print("="*60)
    
    file_path = input("Enter IFC file path (.ifc): ").strip()
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return
    
    print(f"📄 Parsing IFC file: {os.path.basename(file_path)}")
    parser = IFCParser()
    records = parser.parse_ifc(file_path)
    
    if not records:
        print("❌ No elements extracted from IFC file.")
        return
    
    JSONStorage.save(project_id, "L1_ifc.json", records)
    print(f"✅ L1 JSON saved. Total elements: {len(records)}")
    
    # Show sample
    print(f"\n📊 Sample Element:")
    sample = records[0]
    print(f"   ID: {sample.get('id', 'N/A')}")
    print(f"   Type: {sample.get('entity_type', 'N/A')}")
    print(f"   Name: {sample.get('properties', {}).get('Name', 'N/A')}")


# ==================== UPDATED L2 INGESTION ====================
def ingest_l2(project_id):
    """Add product data (L2) using enhanced ProductExtractor"""
    print("\n" + "="*60)
    print("📋 L2 PRODUCT DATA INGESTION (Enhanced Parser)")
    print("="*60)
    print("Supported formats:")
    print("  • Excel (.xlsx, .xls) - With intelligent header detection")
    print("  • PDF (.pdf) - Product catalogs with pattern matching & table extraction")
    print("="*60)
    
    # Initialize extractor
    extractor = ProductExtractor()
    
    file_path = input("\nEnter product file path: ").strip()
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return
    
    # Auto-detect file type
    ext = os.path.splitext(file_path)[1].lower()
    
    products = []
    
    try:
        if ext in ['.xlsx', '.xls', '.xlsm']:
            # Excel file
            print(f"\n📄 Processing Excel file: {os.path.basename(file_path)}")
            
            # Ask for sheet name
            sheet_input = input("Enter sheet name (press Enter for first sheet): ").strip()
            sheet_name = sheet_input if sheet_input else None
            
            products = extractor.extract_from_excel(file_path, sheet_name)
            
        elif ext == '.pdf':
            # PDF file
            print(f"\n📄 Processing PDF file: {os.path.basename(file_path)}")
            products = extractor.extract_from_pdf(file_path)
            
        else:
            print(f"❌ Unsupported file type: {ext}")
            return
        
        if not products:
            print("❌ No products extracted.")
            return
        
        # Save to storage
        JSONStorage.save(project_id, "L2_product.json", products)
        
        # Show statistics
        stats = extractor.get_stats()
        print(f"\n📊 Extraction Statistics:")
        print(f"   Products extracted: {stats['products_extracted']}")
        if ext == '.pdf':
            print(f"   Pattern matches: {stats.get('pdf_pattern_matches', 0)}")
        
        # Display sample
        print(f"\n📊 Sample Product (1 of {len(products)}):")
        print("-" * 40)
        sample = products[0]
        props = sample.get('properties', {})
        
        # Show key fields
        print(f"  Product: {props.get('product_name', 'N/A')}")
        print(f"  Manufacturer: {props.get('manufacturer', 'N/A')}")
        print(f"  Model: {props.get('model_number', 'N/A')}")
        
        # Show technical specs if available
        tech_fields = ['fire_rating_hours', 'compressive_strength_mpa', 'thickness_mm']
        for field in tech_fields:
            if field in props:
                print(f"  {field.replace('_', ' ').title()}: {props[field]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def ingest_l4(project_id):
    """Add regulations (L4)"""
    print("\n" + "="*60)
    print("📜 L4 REGULATION INGESTION")
    print("="*60)
    
    pipeline = L4Pipeline()
    all_records = []
    
    print("Enter PDF file paths (type 'done' to finish):")
    print("-" * 40)
    
    while True:
        file_path = input("PDF path: ").strip()
        if file_path.lower() == "done":
            break
        if not os.path.exists(file_path):
            print("⚠️ File not found. Try again.")
            continue
        
        print(f"📄 Parsing: {os.path.basename(file_path)}")
        records = pipeline.parse(file_path)
        all_records.extend(records)
        print(f"✅ Parsed {len(records)} rules")
    
    if not all_records:
        print("❌ No regulations parsed.")
        return
    
    JSONStorage.save(project_id, "L4_regulation.json", all_records)
    print(f"\n✅ L4 JSON saved. Total regulations: {len(all_records)}")
    
    # Show statistics
    numeric_rules = sum(1 for r in all_records if r.get('properties', {}).get('is_numeric_rule', False))
    print(f"📊 Numeric rules: {numeric_rules}")
    print(f"   General rules: {len(all_records) - numeric_rules}")


# ==================== UPDATED L5 INGESTION ====================
def ingest_l5(project_id):
    """Add project requirements (L5) from Excel or PDF using unified parser"""
    print("\n" + "="*60)
    print("📋 L5 REQUIREMENT INGESTION (Unified Parser)")
    print("="*60)
    print("Supported formats:")
    print("  • Excel (.xlsx, .xls) - With intelligent header detection & noise removal")
    print("  • PDF (.pdf) - Schedule of Rates table extraction")
    print("="*60)
    
    file_path = input("\nEnter file path: ").strip()
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return
    
    # Auto-detect file type
    ext = os.path.splitext(file_path)[1].lower()
    
    # Ask for sheet name only for Excel files
    sheet_name = None
    if ext in ['.xlsx', '.xls', '.xlsm']:
        sheet_input = input("Enter sheet name (press Enter for first sheet): ").strip()
        if sheet_input:
            sheet_name = sheet_input
    
    # Use unified parser
    parser = RequirementParser()
    requirements = parser.parse(
        file_path=file_path,
        sheet_name=sheet_name  # Will be ignored for PDF
    )
    
    if not requirements:
        print("❌ No requirements extracted.")
        return
    
    # Save to storage
    JSONStorage.save(project_id, "L5_requirement.json", requirements)
    
    # Display sample
    print(f"\n📊 Sample Requirement (1 of {len(requirements)}):")
    print("-" * 40)
    sample = requirements[0]
    props = sample.get('properties', {})
    
    print(f"  ID: {props.get('RequirementID', 'N/A')}")
    print(f"  Element Type: {props.get('ElementType', 'N/A')}")
    print(f"  Property: {props.get('Property', 'N/A')}")
    print(f"  Rule: {props.get('Property', 'N/A')} {props.get('Operator', 'N/A')} {props.get('RequiredValue', 'N/A')} {props.get('Unit', 'N/A')}")
    print(f"  Priority: {props.get('Priority', 'N/A')}")
    
    # Show statistics if available from parser
    if hasattr(parser, 'stats') and parser.stats.get('valid_requirements', 0) > 0:
        print(f"\n📊 Parser Statistics:")
        print(f"  Total rows scanned: {parser.stats.get('total_rows', 0)}")
        print(f"  Noise rows removed: {parser.stats.get('noise_removed', 0)}")
        print(f"  Empty rows removed: {parser.stats.get('empty_removed', 0)}")
        if ext == '.pdf':
            print(f"  Tables found: {parser.stats.get('pdf_tables_found', 0)}")
            print(f"  Rows extracted: {parser.stats.get('pdf_rows_extracted', 0)}")


# ==================== Compliance Functions ====================

def run_compliance_l1_l2_l4(project_id):
    """Run L1+L2+L4 compliance check"""
    print("\n" + "="*60)
    print("🔍 L1-L2-L4 REGULATORY COMPLIANCE CHECK")
    print("="*60)
    
    l1 = JSONStorage.load(project_id, "L1_ifc.json")
    l2 = JSONStorage.load(project_id, "L2_product.json")
    l4 = JSONStorage.load(project_id, "L4_regulation.json")
    
    if not l1:
        print("❌ Missing L1 data. Please ingest IFC model first.")
        return
    if not l2:
        print("❌ Missing L2 data. Please ingest product data first.")
        return
    if not l4:
        print("❌ Missing L4 data. Please ingest regulations first.")
        return
    
    print(f"📊 Loaded: {len(l1)} elements, {len(l2)} products, {len(l4)} regulations")
    
    engine = ComplianceEngine(l1, l2, l4)
    mismatches = engine.run()
    
    if mismatches:
        JSONStorage.save(project_id, "mismatch.json", mismatches)
        print(f"\n✅ Compliance check complete. Found {len(mismatches)} issues.")
        
        # Show summary
        by_type = {}
        for m in mismatches:
            issue_type = m.get('issue_type', 'Unknown')
            by_type[issue_type] = by_type.get(issue_type, 0) + 1
        
        print("\n📊 Issue Summary:")
        for issue_type, count in by_type.items():
            print(f"  • {issue_type}: {count}")
    else:
        print("\n✅ No compliance issues found! All elements meet regulations.")


def run_compliance_l1_l2_l5(project_id):
    """Run L1+L2+L5 compliance check"""
    print("\n" + "="*60)
    print("🔍 L1-L2-L5 PROJECT COMPLIANCE CHECK")
    print("="*60)
    
    l1 = JSONStorage.load(project_id, "L1_ifc.json")
    l5 = JSONStorage.load(project_id, "L5_requirement.json")
    
    if not l1:
        print("❌ Missing L1 data. Please ingest IFC model first.")
        return
    if not l5:
        print("❌ Missing L5 data. Please ingest requirements first.")
        return
    
    print(f"📊 Loaded: {len(l1)} elements, {len(l5)} requirements")
    
    engine = ComplianceEngineL5(l1, l5)
    mismatches = engine.run()
    
    if mismatches:
        JSONStorage.save(project_id, "mismatch_l5.json", mismatches)
        
        # Also append to main mismatch
        existing = JSONStorage.load(project_id, "mismatch.json") or []
        existing.extend(mismatches)
        JSONStorage.save(project_id, "mismatch.json", existing)
        
        print(f"\n✅ Saved {len(mismatches)} issues to mismatch_l5.json")
        
        # Show priority breakdown
        priorities = {'High': 0, 'Medium': 0, 'Low': 0}
        for m in mismatches:
            pri = m.get('priority', 'Medium')
            priorities[pri] = priorities.get(pri, 0) + 1
        
        print("\n📊 Priority Breakdown:")
        for pri, count in priorities.items():
            if count > 0:
                print(f"  • {pri}: {count}")
    else:
        print("\n✅ No compliance issues found! All elements meet requirements.")


def run_compliance_l4_l5(project_id):
    """Run L4-L5 compliance check comparing regulations with requirements"""
    print("\n" + "="*60)
    print("🔍 L4-L5 REGULATION VS REQUIREMENT COMPLIANCE CHECK")
    print("="*60)
    
    l4 = JSONStorage.load(project_id, "L4_regulation.json")
    l5 = JSONStorage.load(project_id, "L5_requirement.json")
    
    if not l4:
        print("❌ Missing L4 data. Please ingest regulations first.")
        return
    if not l5:
        print("❌ Missing L5 data. Please ingest requirements first.")
        return
    
    print(f"📊 Loaded: {len(l4)} regulations, {len(l5)} requirements")
    
    engine = ComplianceEngineL4L5(l4, l5)
    comparisons = engine.run()
    
    if comparisons:
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"l4_l5_comparison_{timestamp}.json"
        JSONStorage.save(project_id, filename, comparisons)
        print(f"\n✅ Saved {len(comparisons)} L4-L5 comparisons to {filename}")
        
        # Also save as latest
        JSONStorage.save(project_id, "l4_l5_latest.json", comparisons)
        
        # Show relationship summary
        relationships = {}
        for comp in comparisons:
            rel = comp.get('relationship', 'unknown')
            relationships[rel] = relationships.get(rel, 0) + 1
        
        print("\n📊 Relationship Summary:")
        rel_icons = {
            'equal': '✅', 'stricter': '🔼', 
            'weaker': '🔽', 'missing': '❌',
            'incomparable': '⚠️'
        }
        for rel, count in relationships.items():
            icon = rel_icons.get(rel, '•')
            print(f"  {icon} {rel.title()}: {count}")
    else:
        print("\n✅ No compliance issues found! All requirements align with regulations.")


# ==================== Vector Store Functions ====================

def build_vector_store(project_id):
    """Build unified vector store"""
    print("\n" + "="*60)
    print("📚 BUILDING UNIFIED VECTOR STORE")
    print("="*60)
    
    store = UnifiedVectorStore()
    store.build_from_project(project_id)
    
    path = f"data/processed/{project_id}/unified.index"
    store.save(path)
    print(f"\n✅ Vector store saved to {path}")
    
    # Show stats
    if hasattr(store, 'text_chunks'):
        print(f"📊 Total chunks indexed: {len(store.text_chunks)}")


# ==================== Query Functions ====================

def query_knowledge_base(project_id):
    """Query using the router system"""
    print("\n" + "="*60)
    print("🔍 KNOWLEDGE BASE QUERY")
    print("="*60)
    
    query = input("\nEnter your question: ").strip()
    if not query:
        return
    
    print(f"\n📤 Processing query: '{query}'")
    
    # Initialize router
    router = QueryRouter(project_id)
    
    # Get results
    result = router.retrieve(query, top_k=5)
    
    # Display routing info
    print(f"\n🔍 Query routed to: {result['route']['retriever']}")
    print(f"   Confidence: {result['route']['confidence']}")
    print(f"   Method: {result['route']['details'].get('method', 'semantic')}")
    
    # Display results
    print(f"\n📋 Found {result['total_found']} results:\n")
    
    for i, res in enumerate(result['results'], 1):
        print(f"{'='*80}")
        print(f"Result {i} (Score: {res['relevance_score']:.2f})")
        print(f"Layer: {res['layer']} | Source: {res['source']}")
        print(f"Content: {res['content']}")
        
        # Show key metadata
        metadata = res.get('metadata', {})
        if metadata:
            important_keys = []
            
            # L1 metadata
            if res['layer'] in [1, '1']:
                important_keys = ['id', 'name', 'type']
            
            # L2 metadata
            elif res['layer'] in [2, '2']:
                important_keys = ['product_name', 'manufacturer', 'model_number', 'fire_rating_hours']
            
            # L4 metadata
            elif res['layer'] in [4, '4']:
                important_keys = ['rule_type', 'value', 'unit']
            
            # L5 metadata
            elif res['layer'] in [5, '5']:
                important_keys = ['RequirementID', 'Property', 'RequiredValue', 'Unit', 'Priority']
            
            # Compliance metadata
            elif res['layer'] == 'compliance':
                important_keys = ['relationship', 'priority', 'element_name', 'requirement_id']
            
            # Show important keys
            shown = False
            for key in important_keys:
                if key in metadata:
                    if not shown:
                        print("  Metadata:")
                        shown = True
                    print(f"    {key}: {metadata[key]}")


def view_files(project_id):
    """View stored JSON files with details"""
    project_path = os.path.join(PROJECTS_PATH, project_id)
    files = os.listdir(project_path)
    
    if not files:
        print("\n📁 No JSON files stored yet.")
        return
    
    print("\n📁 Stored JSON Files:")
    print("-" * 60)
    print(f"{'Filename':<30} {'Size':>10} {'Records':>10}")
    print("-" * 60)
    
    for f in sorted(files):
        if f.endswith('.json'):
            file_path = os.path.join(project_path, f)
            size = os.path.getsize(file_path)
            
            # Try to count records
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    records = len(data) if isinstance(data, list) else 1
            except:
                records = '?'
            
            print(f"{f:<30} {size:>10,} bytes {records:>10}")


# ==================== Main Menu ====================

def main():
    """Main entry point"""
    project_id = select_project()
    
    while True:
        print(f"\n{'='*60}")
        print(f"📁 PROJECT: {project_id}")
        print(f"{'='*60}")
        print("1. Add IFC Model (L1)")
        print("2. Add Product Data (L2) - Excel/PDF (Enhanced Parser)")
        print("3. Add Regulation (L4)")
        print("4. Add Requirements (L5) - Excel/PDF (Unified Parser)")
        print("5. View Stored Files")
        print("6. Run L1-L2-L4 Compliance (Regulations)")
        print("7. Run L1-L2-L5 Compliance (Requirements)")
        print("8. Run L4-L5 Compliance (Regulation vs Requirement)")
        print("9. Build Vector Store")
        print("10. Query Knowledge Base")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            ingest_l1(project_id)
        elif choice == "2":
            ingest_l2(project_id)  # Updated with enhanced parser
        elif choice == "3":
            ingest_l4(project_id)
        elif choice == "4":
            ingest_l5(project_id)
        elif choice == "5":
            view_files(project_id)
        elif choice == "6":
            run_compliance_l1_l2_l4(project_id)
        elif choice == "7":
            run_compliance_l1_l2_l5(project_id)
        elif choice == "8":
            run_compliance_l4_l5(project_id)
        elif choice == "9":
            build_vector_store(project_id)
        elif choice == "10":
            query_knowledge_base(project_id)
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice.")


if __name__ == "__main__":
    main()