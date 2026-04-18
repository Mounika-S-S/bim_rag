import os
import json
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.core.json_storage import JSONStorage
from src.inference.compliance_engine import ComplianceEngine

def main():
    project_id = "project-final"
    
    print(f"Loading data for project: {project_id}")
    l1 = JSONStorage.load(project_id, "L1_ifc.json") or []
    l2 = JSONStorage.load(project_id, "L2_product.json") or []
    l4 = JSONStorage.load(project_id, "L4_regulation.json") or []
    l5 = JSONStorage.load(project_id, "L5_requirement.json") or []

    print(f"L1: {len(l1)} records")
    print(f"L2: {len(l2)} records")
    print(f"L4: {len(l4)} records")
    print(f"L5: {len(l5)} records")

    if not l1:
        print("Missing L1 (IFC) records. Cannot run deterministic inference.")
        return

    print("\nRunning Compliance Engine...")
    engine = ComplianceEngine(l1, l2, l4, l5)
    results = engine.run()

    print(f"Compliance Inference completed. Total results: {len(results)}")
    
    # Search for an element that we know has properties
    target_id = "1FFFWQ3Nn7SQRMK$3B1XHi"
    target_results = [r for r in results if r.get("element_id") == target_id]
    
    print(f"\nResults for element {target_id}:")
    for r in target_results:
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
