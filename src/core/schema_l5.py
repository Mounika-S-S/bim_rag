# src/core/schema_l5.py
"""
Extended schema for L1-L2-L5 compliance checking
Based on ontology-based compliance frameworks [citation:3][citation:5]
"""
from src.core.schema import create_layer_record
from src.utils.id_generator import generate_id

def create_compliance_record(element, requirement, comparison_result):
    """
    Creates a standardized compliance check record
    
    Args:
        element: L1 IFC element
        requirement: L5 requirement
        comparison_result: Dict with comparison details
    
    Returns:
        Standardized compliance record
    """
    element_props = element.get('properties', {})
    req_props = requirement.get('properties', {})
    
    return {
        "id": generate_id("COMP"),
        "element_id": element.get('id'),
        "element_name": element_props.get('Name', 'Unknown'),
        "element_type": element.get('entity_type', ''),
        "requirement_id": req_props.get('RequirementID', 'Unknown'),
        "requirement_text": req_props.get('Description', ''),
        "rule": f"{req_props.get('Property', '')} {req_props.get('Operator', '')} {req_props.get('RequiredValue', '')} {req_props.get('Unit', '')}",
        "actual_value": comparison_result.get('actual_value'),
        "required_value": comparison_result.get('required_value'),
        "comparison": comparison_result.get('comparison'),
        "is_compliant": comparison_result.get('is_compliant', False),
        "priority": req_props.get('Priority', 'Medium'),
        "layer_check": "L1_L2_L5",
        "timestamp": None  # Add datetime.now() if needed
    }