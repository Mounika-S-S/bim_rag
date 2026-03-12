# src/core/schema_l4_l5.py
"""
Extended schema for L4-L5 compliance checking (Regulations vs Requirements)
Based on semantic interoperability research [citation:1] where heterogeneous
data sources are formally specified and verified against regulations.

This schema captures the relationship between building codes (L4) and 
project requirements (L5), enabling comparison of:
- Stricter requirements (project exceeds code)
- Gaps (code requirements missing from project)
- Conflicts (rules that disagree with regulations)
"""
from src.core.schema import create_layer_record
from src.utils.id_generator import generate_id
from datetime import datetime


def create_l4_l5_compliance_record(l4_rule, l5_requirement, comparison_result):
    """
    Creates a standardized compliance record comparing L4 and L5
    
    Args:
        l4_rule: Regulation from L4_regulation.json
        l5_requirement: Requirement from L5_requirement.json
        comparison_result: Dict with comparison details
    
    Returns:
        Standardized compliance record for L4-L5 comparison
    """
    l4_props = l4_rule.get('properties', {})
    l5_props = l5_requirement.get('properties', {})
    
    # Extract comparable values
    l4_value = l4_props.get('threshold_value')
    l4_unit = l4_props.get('unit', '')
    l4_operator = l4_props.get('comparison_operator', '>=')
    l4_rule_type = l4_props.get('rule_type', 'General')
    l4_text = l4_props.get('text', '')
    
    l5_value = l5_props.get('RequiredValue')
    l5_unit = l5_props.get('Unit', '')
    l5_operator = l5_props.get('Operator', '>=')
    l5_rule_type = l5_props.get('Property', 'General')
    l5_text = l5_props.get('Description', '')
    l5_priority = l5_props.get('Priority', 'Medium')
    
    # Determine relationship type
    relationship = _determine_relationship(
        l4_value, l4_operator, l5_value, l5_operator,
        l4_unit, l5_unit
    )
    
    # FIXED: Removed len() and fixed string slicing
    l4_id_prefix = str(l4_rule.get('id', ''))[:4] if l4_rule.get('id') else 'xxxx'
    l5_id_prefix = str(l5_requirement.get('id', ''))[:4] if l5_requirement.get('id') else 'xxxx'
    
    return {
        "id": generate_id("L4L5"),
        "comparison_id": f"L4L5_{l4_id_prefix}_{l5_id_prefix}",
        "l4_rule": {
            "id": l4_rule.get('id'),
            "rule_type": l4_rule_type,
            "text": l4_text,
            "value": l4_value,
            "operator": l4_operator,
            "unit": l4_unit
        },
        "l5_requirement": {
            "id": l5_requirement.get('id'),
            "requirement_id": l5_props.get('RequirementID'),
            "property": l5_rule_type,
            "text": l5_text,
            "value": l5_value,
            "operator": l5_operator,
            "unit": l5_unit,
            "priority": l5_priority
        },
        "comparison": comparison_result.get('comparison', ''),
        "relationship": relationship,
        "is_compliant": comparison_result.get('is_compliant', True),
        "gap_description": _generate_gap_description(
            l4_text, l5_text, relationship, 
            l4_value, l4_unit, l5_value, l5_unit
        ),
        "layer_check": "L4_L5",
        "timestamp": datetime.now().isoformat()
    }


def _determine_relationship(l4_val, l4_op, l5_val, l5_op, l4_unit, l5_unit):
    """
    Determine relationship between L4 and L5 values
    Returns: 'stricter', 'weaker', 'equal', 'incomparable'
    """
    # Convert to numbers if possible
    try:
        l4_num = float(l4_val) if l4_val is not None else None
        l5_num = float(l5_val) if l5_val is not None else None
    except (ValueError, TypeError):
        return 'incomparable'
    
    if l4_num is None or l5_num is None:
        return 'incomparable'
    
    # Unit conversion (simplified - expand as needed)
    if l4_unit != l5_unit:
        if l4_unit == 'm' and l5_unit == 'mm':
            l5_num = l5_num / 1000
        elif l4_unit == 'mm' and l5_unit == 'm':
            l5_num = l5_num * 1000
        # Add more unit conversions as needed
    
    # Compare based on operators
    if l4_op == '>=' and l5_op == '>=':
        if l5_num > l4_num:
            return 'stricter'  # Project requires more than code
        elif l5_num < l4_num:
            return 'weaker'    # Project allows less than code
        else:
            return 'equal'
    
    elif l4_op == '<=' and l5_op == '<=':
        if l5_num < l4_num:
            return 'stricter'  # Project allows less (tighter)
        elif l5_num > l4_num:
            return 'weaker'    # Project allows more (looser)
        else:
            return 'equal'
    
    # Default
    return 'incomparable'


def _generate_gap_description(l4_text, l5_text, relationship, l4_val, l4_unit, l5_val, l5_unit):
    """Generate human-readable gap description"""
    if relationship == 'stricter':
        return f"Project requirement is STRICTER than code: L5 requires {l5_val}{l5_unit} vs L4 requires {l4_val}{l4_unit}"
    elif relationship == 'weaker':
        return f"Project requirement is WEAKER than code: L5 allows {l5_val}{l5_unit} but L4 requires {l4_val}{l4_unit}"
    elif relationship == 'equal':
        return f"Project requirement matches code: both require {l4_val}{l4_unit}"
    else:
        return f"Gap between regulation and requirement: cannot directly compare"