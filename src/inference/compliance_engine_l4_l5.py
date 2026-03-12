# src/inference/compliance_engine_l4_l5.py
"""
L4-L5 Compliance Engine
Compares Building Regulations (L4) against Project Requirements (L5)
Based on semantic interoperability research [citation:1] for verifying
data against different regulations and requirements.

Features:
- Identifies stricter requirements (project exceeds code)
- Identifies gaps (code requirements missing from project)
- Identifies conflicts (rules that disagree)
- Generates compliance reports
"""
import re
from src.core.json_storage import JSONStorage
from src.core.schema_l4_l5 import create_l4_l5_compliance_record
from src.utils.id_generator import generate_id
from datetime import datetime

class ComplianceEngineL4L5:
    """
    Compares regulations (L4) with project requirements (L5)
    Strategy:
    1. Match L4 rules to L5 requirements by topic (fire, height, setback, etc.)
    2. Compare threshold values and operators
    3. Classify relationship: stricter, weaker, equal, missing, extra
    4. Generate comprehensive gap analysis
    """
    
    # Topic mapping between L4 rule types and L5 properties
    TOPIC_MAPPING = {
        'fire': ['FireRating', 'Fire', 'fire_rating'],
        'height': ['Height', 'BuildingHeight', 'MaxHeight'],
        'setback': ['Setback', 'FrontSetback', 'SideSetback'],
        'parking': ['Parking', 'ParkingSpaces'],
        'thickness': ['Thickness', 'WallThickness'],
        'width': ['Width', 'CorridorWidth', 'DoorWidth'],
        'area': ['Area', 'FloorArea', 'RoomArea'],
        'fsi': ['FSI', 'FloorSpaceIndex'],
        'stair': ['StairWidth', 'StairRiser'],
        'sanitary': ['WaterCloset', 'Sanitary', 'Toilet']
    }
    
    def __init__(self, l4_records, l5_records):
        """
        Initialize with L4 and L5 data
        
        Args:
            l4_records: List of regulations from L4_regulation.json
            l5_records: List of requirements from L5_requirement.json
        """
        self.l4 = l4_records
        self.l5 = l5_records
        self.comparisons = []
        
        # Statistics
        self.stats = {
            "l4_rules": len(l4_records),
            "l5_requirements": len(l5_records),
            "matches": 0,
            "stricter": 0,    # Project stricter than code
            "weaker": 0,       # Project weaker than code
            "equal": 0,        # Project matches code
            "missing": 0,      # Code requirement missing from project
            "extra": 0,        # Project requirement beyond code
            "incomparable": 0
        }
    
    def run(self):
        """
        Run compliance check comparing L4 regulations with L5 requirements
        """
        print("\n" + "="*60)
        print("🔍 L4-L5 COMPLIANCE CHECK")
        print("="*60)
        print("Comparing Building Regulations (L4) with Project Requirements (L5)")
        print("="*60)
        
        # Group L5 requirements by topic
        l5_by_topic = self._group_l5_by_topic()
        
        # Compare each L4 rule with matching L5 requirements
        for rule in self.l4:
            self._compare_rule_with_requirements(rule, l5_by_topic)
        
        # Identify missing requirements (code requirements not in project)
        self._identify_missing_requirements(l5_by_topic)
        
        # Print summary
        self._print_summary()
        
        return self.comparisons
    
    def _group_l5_by_topic(self):
        """
        Group L5 requirements by their topic/property
        """
        l5_by_topic = {}
        
        for req in self.l5:
            props = req.get('properties', {})
            property_name = props.get('Property', '').lower()
            
            # Find matching topic
            matched_topic = None
            for topic, keywords in self.TOPIC_MAPPING.items():
                if any(keyword.lower() in property_name for keyword in keywords):
                    matched_topic = topic
                    break
            
            if matched_topic:
                if matched_topic not in l5_by_topic:
                    l5_by_topic[matched_topic] = []
                l5_by_topic[matched_topic].append(req)
            else:
                # Uncategorized requirements
                if 'other' not in l5_by_topic:
                    l5_by_topic['other'] = []
                l5_by_topic['other'].append(req)
        
        return l5_by_topic
    
    def _compare_rule_with_requirements(self, rule, l5_by_topic):
        """Compare a single L4 rule with matching L5 requirements"""
        props = rule.get('properties', {})
        rule_text = props.get('text', '').lower()
        rule_type = props.get('rule_type', '').lower()
        rule_value = props.get('threshold_value')
        rule_operator = props.get('comparison_operator', '>=')
        rule_unit = props.get('unit', '')
        
        # Skip non-numeric rules
        if not props.get('is_numeric_rule', False) or rule_value is None:
            return
        
        # Find matching topic
        matched_topic = None
        for topic, keywords in self.TOPIC_MAPPING.items():
            if topic in rule_type or any(keyword.lower() in rule_text for keyword in keywords):
                matched_topic = topic
                break
        
        # TRACK COMPARISONS TO AVOID DUPLICATES
        compared_ids = set()
        
        if matched_topic and matched_topic in l5_by_topic:
            # Compare with all matching requirements
            for req in l5_by_topic[matched_topic]:
                # Create a unique key to avoid duplicates
                comp_key = f"{rule.get('id')}_{req.get('id')}"
                if comp_key in compared_ids:
                    continue
                compared_ids.add(comp_key)
                
                self._compare_single(rule, req)
                self.stats['matches'] += 1
        else:
            # No matching requirement found - this is a gap
            self.stats['missing'] += 1
            self._create_gap_record(rule, None, {
                'comparison': 'No matching requirement found',
                'is_compliant': False,
                'relationship': 'missing'
            })
    
    def _compare_single(self, l4_rule, l5_requirement):
        """
        Compare a single L4 rule with a single L5 requirement
        """
        l4_props = l4_rule.get('properties', {})
        l5_props = l5_requirement.get('properties', {})
        
        l4_value = l4_props.get('threshold_value')
        l5_value = l5_props.get('RequiredValue')
        l4_operator = l4_props.get('comparison_operator', '>=')
        l5_operator = l5_props.get('Operator', '>=')
        l4_unit = l4_props.get('unit', '')
        l5_unit = l5_props.get('Unit', '')
        
        # Convert to numbers for comparison
        try:
            l4_num = float(l4_value) if l4_value is not None else None
            l5_num = float(l5_value) if l5_value is not None else None
        except (ValueError, TypeError):
            self._create_gap_record(l4_rule, l5_requirement, {
                'comparison': 'Non-numeric values cannot be compared',
                'is_compliant': False,
                'relationship': 'incomparable'
            })
            self.stats['incomparable'] += 1
            return
        
        if l4_num is None or l5_num is None:
            self.stats['incomparable'] += 1
            return
        
        # Unit conversion
        l5_converted = self._convert_unit(l5_num, l5_unit, l4_unit)
        
        # Determine relationship
        relationship = self._determine_relationship(
            l4_num, l4_operator, l5_converted, l5_operator
        )
        
        # Update stats
        if relationship == 'stricter':
            self.stats['stricter'] += 1
        elif relationship == 'weaker':
            self.stats['weaker'] += 1
        elif relationship == 'equal':
            self.stats['equal'] += 1
        else:
            self.stats['incomparable'] += 1
        
        # Create comparison record
        comparison_result = {
            'comparison': f"L4: {l4_operator} {l4_num}{l4_unit} vs L5: {l5_operator} {l5_num}{l5_unit}",
            'is_compliant': relationship in ['equal', 'stricter'],
            'relationship': relationship
        }
        
        record = create_l4_l5_compliance_record(
            l4_rule, l5_requirement, comparison_result
        )
        self.comparisons.append(record)
        
        # Print for debugging
        status = "✅" if relationship in ['equal', 'stricter'] else "⚠️"
        print(f"   {status} {l4_props.get('rule_type', 'Rule')} vs {l5_props.get('RequirementID', 'Req')}: {relationship}")
    
    def _determine_relationship(self, l4_val, l4_op, l5_val, l5_op):
        """
        Determine if project requirement is stricter, weaker, or equal to code
        """
        if l4_op == '>=' and l5_op == '>=':
            if l5_val > l4_val:
                return 'stricter'
            elif l5_val < l4_val:
                return 'weaker'
            else:
                return 'equal'
        
        elif l4_op == '<=' and l5_op == '<=':
            if l5_val < l4_val:
                return 'stricter'
            elif l5_val > l4_val:
                return 'weaker'
            else:
                return 'equal'
        
        # Different operators - more complex logic needed
        # This is simplified - expand based on your needs
        return 'incomparable'
    
    def _convert_unit(self, value, from_unit, to_unit):
        """Convert between units"""
        if from_unit == to_unit:
            return value
        
        # Length conversions
        if from_unit == 'm' and to_unit == 'mm':
            return value * 1000
        if from_unit == 'mm' and to_unit == 'm':
            return value / 1000
        if from_unit == 'cm' and to_unit == 'mm':
            return value * 10
        if from_unit == 'mm' and to_unit == 'cm':
            return value / 10
        
        # If no conversion available, return original
        return value
    
    def _create_gap_record(self, l4_rule, l5_requirement, comparison_result):
        """Create a record for a gap (missing requirement)"""
        if l5_requirement is None:
            # L4 rule with no matching L5 requirement
            l4_props = l4_rule.get('properties', {})
            
            record = {
                "id": generate_id("GAP"),
                "comparison_id": f"GAP_{l4_props.get('rule_type', 'Unknown')}",
                "l4_rule": {
                    "id": l4_rule.get('id'),
                    "rule_type": l4_props.get('rule_type', 'General'),
                    "text": l4_props.get('text', ''),
                    "value": l4_props.get('threshold_value'),
                    "operator": l4_props.get('comparison_operator'),
                    "unit": l4_props.get('unit')
                },
                "l5_requirement": None,
                "comparison": "Code requirement not addressed in project specifications",
                "relationship": "missing",
                "is_compliant": False,
                "gap_description": f"Missing requirement: {l4_props.get('text', '')}",
                "layer_check": "L4_L5",
                "timestamp": datetime.now().isoformat()
            }
            self.comparisons.append(record)
    
    def _identify_missing_requirements(self, l5_by_topic):
        """Identify L5 requirements that have no corresponding L4 rule"""
        # This would track extra project requirements beyond code
        # Implementation depends on your needs
        pass
    
    def _print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*60)
        print("📊 L4-L5 COMPLIANCE SUMMARY")
        print("="*60)
        print(f"L4 Regulations analyzed: {self.stats['l4_rules']}")
        print(f"L5 Requirements analyzed: {self.stats['l5_requirements']}")
        print(f"Comparisons made: {self.stats['matches']}")
        print("\nRelationship Analysis:")
        print(f"  ✅ Equal (matches code): {self.stats['equal']}")
        print(f"  🔼 Stricter (exceeds code): {self.stats['stricter']}")
        print(f"  🔽 Weaker (below code): {self.stats['weaker']}")
        print(f"  ❌ Missing from project: {self.stats['missing']}")
        print(f"  ⚠️ Incomparable: {self.stats['incomparable']}")
        
        # Calculate compliance score
        total_comparable = self.stats['equal'] + self.stats['stricter'] + self.stats['weaker']
        if total_comparable > 0:
            compliant = self.stats['equal'] + self.stats['stricter']
            score = (compliant / total_comparable) * 100
            print(f"\n📈 Compliance Score: {score:.1f}%")