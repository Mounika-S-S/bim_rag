# src/inference/compliance_engine_l5.py
"""
L1-L2-L5 Compliance Engine
Compares IFC elements (L1) against Project Requirements (L5)
Uses a hybrid approach: keyword-based matching + rule-based comparison
"""
import re
from src.core.json_storage import JSONStorage


class ComplianceEngineL5:
    """
    Checks compliance between IFC elements and project requirements
    Strategy:
    1. Match IFC elements to requirements based on element type
    2. Extract property values from IFC
    3. Compare against requirement thresholds
    4. Generate mismatch records
    """
    
    def __init__(self, l1_records, l5_records):
        """
        Initialize with L1 and L5 data
        
        Args:
            l1_records: List of IFC elements from L1_ifc.json
            l5_records: List of requirements from L5_requirement.json
        """
        self.l1 = l1_records
        self.l5 = l5_records
        self.mismatches = []
        
        # Statistics
        self.stats = {
            "elements_checked": 0,
            "requirements_applied": 0,
            "violations": 0,
            "missing_properties": 0
        }
    
    def run(self):
        """
        Run compliance check for all L5 requirements against L1 elements
        """
        print("\n" + "="*60)
        print("🔍 L1-L5 COMPLIANCE CHECK")
        print("="*60)
        
        # Group requirements by element type
        req_by_type = self._group_requirements()
        print(f"\n📊 Requirements by type:")
        for etype, reqs in req_by_type.items():
            print(f"   {etype}: {len(reqs)} requirements")
        
        # Check each element against applicable requirements
        for element in self.l1:
            self._check_element(element, req_by_type)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 COMPLIANCE SUMMARY")
        print("="*60)
        print(f"Elements checked: {self.stats['elements_checked']}")
        print(f"Requirements applied: {self.stats['requirements_applied']}")
        print(f"Violations found: {self.stats['violations']}")
        print(f"Missing properties: {self.stats['missing_properties']}")
        
        return self.mismatches
    
    def _group_requirements(self):
        """
        Group requirements by the element type they apply to
        """
        req_by_type = {}
        
        for req in self.l5:
            props = req.get('properties', {})
            
            # Get element type from requirement
            element_type = props.get('ElementType', '')
            
            # Handle wildcard or empty (apply to all)
            if not element_type or element_type == 'All':
                element_type = 'IfcElement'  # Generic type
            
            if element_type not in req_by_type:
                req_by_type[element_type] = []
            req_by_type[element_type].append(req)
        
        return req_by_type
    
    def _check_element(self, element, req_by_type):
        """
        Check a single element against applicable requirements
        """
        element_type = element.get('entity_type', '')
        element_props = element.get('properties', {})
        element_id = element.get('id', '')
        element_name = element_props.get('Name', 'Unknown')
        
        self.stats['elements_checked'] += 1
        
        # Find requirements for this element type
        applicable_reqs = []
        
        # Direct type match
        if element_type in req_by_type:
            applicable_reqs.extend(req_by_type[element_type])
        
        # Generic IfcElement requirements
        if 'IfcElement' in req_by_type:
            applicable_reqs.extend(req_by_type['IfcElement'])
        
        if not applicable_reqs:
            return
        
        # Check each requirement
        for req in applicable_reqs:
            self.stats['requirements_applied'] += 1
            self._check_single_requirement(element, req)
    
    def _check_single_requirement(self, element, requirement):
        """
        Check a single requirement against an element
        """
        req_props = requirement.get('properties', {})
        element_props = element.get('properties', {})
        
        # Extract requirement details
        property_name = req_props.get('Property', '')
        operator = req_props.get('Operator', '')
        required_value = req_props.get('RequiredValue', '')
        unit = req_props.get('Unit', '')
        priority = req_props.get('Priority', 'Medium')
        req_id = req_props.get('RequirementID', 'Unknown')
        description = req_props.get('Description', '')
        
        # Skip if no property specified
        if not property_name:
            return
        
        # Get actual value from element
        actual_value = self._extract_property_value(element_props, property_name)
        
        # If property not found
        if actual_value is None:
            self.stats['missing_properties'] += 1
            self._add_mismatch(
                element, requirement,
                f"Missing required property: {property_name}",
                priority,
                actual_value,
                required_value
            )
            return
        
        # Compare values
        is_compliant, comparison_result = self._compare_values(
            actual_value, operator, required_value
        )
        
        if not is_compliant:
            self.stats['violations'] += 1
            self._add_mismatch(
                element, requirement,
                comparison_result,
                priority,
                actual_value,
                required_value
            )
    
    def _extract_property_value(self, properties, property_name):
        """
        Extract property value from element properties
        Handles different naming conventions and nested properties
        """
        # Direct match
        if property_name in properties:
            return properties[property_name]
        
        # Case-insensitive match
        prop_lower = property_name.lower()
        for key, value in properties.items():
            if key.lower() == prop_lower:
                return value
        
        # Check common BIM property variations
        variations = {
            "FireRating": ["FireRating", "Fire_Rating", "fire_rating", "Fire Rating"],
            "IsExternal": ["IsExternal", "External", "is_external"],
            "Height": ["Height", "height", "OverallHeight", "UnconnectedHeight"],
            "Width": ["Width", "width", "OverallWidth"],
            "Thickness": ["Thickness", "thickness", "Width"],
            "Material": ["Material", "material", "Construction Type"],
        }
        
        for std_name, variants in variations.items():
            if property_name == std_name or property_name in variants:
                for variant in variants:
                    if variant in properties:
                        return properties[variant]
        
        return None
    
    def _compare_values(self, actual, operator, required):
        """
        Compare actual value against required value based on operator
        
        Returns:
            (is_compliant, message)
        """
        # Convert to numbers if possible
        actual_num = self._to_number(actual)
        required_num = self._to_number(required)
        
        # If both are numbers, do numeric comparison
        if actual_num is not None and required_num is not None:
            if operator == '>=':
                return actual_num >= required_num, f"{actual_num} >= {required_num}"
            elif operator == '<=':
                return actual_num <= required_num, f"{actual_num} <= {required_num}"
            elif operator == '>':
                return actual_num > required_num, f"{actual_num} > {required_num}"
            elif operator == '<':
                return actual_num < required_num, f"{actual_num} < {required_num}"
            elif operator == '=' or operator == '==':
                return actual_num == required_num, f"{actual_num} == {required_num}"
            elif operator == '!=':
                return actual_num != required_num, f"{actual_num} != {required_num}"
        
        # String comparison
        actual_str = str(actual).strip().lower()
        required_str = str(required).strip().lower()
        
        if operator == 'IN' or operator == 'in':
            # Check if actual is in comma-separated list
            allowed = [x.strip().lower() for x in required_str.split(',')]
            return actual_str in allowed, f"{actual_str} in {allowed}"
        
        if operator == '=' or operator == '==':
            return actual_str == required_str, f"{actual_str} == {required_str}"
        
        if operator == 'CONTAINS' or operator == 'contains':
            return required_str in actual_str, f"{required_str} in {actual_str}"
        
        # Default to equality
        return actual_str == required_str, f"{actual_str} == {required_str}"
    
    def _to_number(self, value):
        """
        Convert value to number if possible
        """
        if isinstance(value, (int, float)):
            return value
        
        if isinstance(value, str):
            # Extract number from string (e.g., "120 min" -> 120)
            match = re.match(r'^(\d+(?:\.\d+)?)', value.strip())
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
        
        return None
    
    def _add_mismatch(self, element, requirement, comparison_result, priority, actual, required):
        """
        Create a mismatch record
        """
        element_props = element.get('properties', {})
        req_props = requirement.get('properties', {})
        
        mismatch = {
            "id": f"L5_mismatch_{len(self.mismatches)}",
            "element_id": element.get('id', ''),
            "element_name": element_props.get('Name', 'Unknown'),
            "element_type": element.get('entity_type', ''),
            "requirement_id": req_props.get('RequirementID', 'Unknown'),
            "requirement_text": req_props.get('Description', ''),
            "rule": f"{req_props.get('Property', '')} {req_props.get('Operator', '')} {req_props.get('RequiredValue', '')} {req_props.get('Unit', '')}",
            "actual_value": actual,
            "required_value": required,
            "comparison": comparison_result,
            "priority": priority,
            "issue": f"Element does not meet requirement: {comparison_result}",
            "layer_check": "L1_L2_L5",
            "timestamp": None  # Could add datetime.now()
        }
        
        self.mismatches.append(mismatch)
        
        # Print for debugging
        print(f"   ⚠️ {mismatch['element_name']}: {comparison_result}")