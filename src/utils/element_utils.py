# src/utils/element_utils.py
"""
Utility functions for element-specific query handling
Centralizes all element pattern logic to avoid code duplication

Based on entity linking research [SIGIR'23] and knowledge graph patterns
"""
import re
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

# COMPLETE list of element patterns from your IFC data
ELEMENT_PATTERNS = [
    # Structural Elements
    "YC-ST-WA-EIP",     # Wall
    "YC-ST-SC-CIP",     # Column
    "YC-ST-SF-BIP",     # Beam/Structural Frame
    "YC-ST-FO-STR",     # Footing/Foundation
    "YC-ST-BM-COM",     # Composite Beam
    
    # Architectural Elements
    "YC-AR-WA-IIP",     # Interior Wall
    "YC-AR-WA-PIP",     # Partition Wall
    "YC-AR-DO-INT",     # Interior Door
    "YC-AR-DO-EXT",     # Exterior Door
    "YC-AR-WN-CAW",     # Casement Window
    "YC-AR-WN-DHW",     # Double Hung Window
    "YC-AR-SD-01",      # Sliding Door
    "YC-AR-FX-01",      # Fixed Window
    
    # MEP Elements
    "YC-MP-DU-01",      # Duct
    "YC-MP-PI-01",      # Pipe
    "YC-MP-AC-01",      # Air Conditioner
    "YC-MP-LT-01",      # Light
    "YC-MP-FN-01",      # Fan
    
    # Openings
    "YC-OP-WN-01",      # Window Opening
    "YC-OP-DR-01",      # Door Opening
    "YC-OP-VT-01",      # Vent Opening
]

# Element type keywords for classification
ELEMENT_TYPES = {
    'wall': ['wall', 'walls', 'partition', 'curtain'],
    'beam': ['beam', 'beams', 'girder', 'joist'],
    'column': ['column', 'columns', 'pillar', 'pier'],
    'slab': ['slab', 'slabs', 'floor', 'deck'],
    'door': ['door', 'doors', 'gate'],
    'window': ['window', 'windows', 'glazing'],
    'footing': ['footing', 'footings', 'foundation', 'pile'],
    'roof': ['roof', 'roofs', 'ceiling'],
    'stair': ['stair', 'stairs', 'staircase', 'ramp'],
    'opening': ['opening', 'openings', 'void', 'hole'],
    'duct': ['duct', 'ducts', 'hvac'],
    'pipe': ['pipe', 'pipes', 'plumbing'],
    'light': ['light', 'lights', 'fixture', 'lamp'],
}

# Phrases that indicate element-specific queries
ELEMENT_QUERY_PHRASES = [
    "issues for", 
    "problems with", 
    "what's wrong with",
    "show issues", 
    "tell me about", 
    "compliance for",
    "non-compliant", 
    "violation for",
    "check compliance of",
    "find element",
    "locate",
    "where is",
    "show me all",
    "list all",
    "display all",
    "get all",
]

# Priority keywords for scoring
PRIORITY_KEYWORDS = {
    "high": ["high", "critical", "urgent", "important"],
    "medium": ["medium", "moderate"],
    "low": ["low", "minor", "optional"]
}

# ==================== ELEMENT EXTRACTION ====================

def extract_element_from_query(query: str) -> Tuple[Optional[str], float]:
    """
    Extract element ID/name from query
    
    Args:
        query: Natural language query string
    
    Returns:
        tuple: (element_name, confidence_score)
        
    Based on entity linking techniques [SIGIR'23]
    """
    if not query:
        return None, 0.0
    
    query_upper = query.upper()
    query_lower = query.lower()
    
    # Strategy 1: Direct pattern match from known elements
    for pattern in ELEMENT_PATTERNS:
        if pattern in query_upper:
            logger.debug(f"   Direct pattern match: {pattern}")
            return pattern, 1.0
    
    # Strategy 2: Look for pattern like YC-ST-WA-EIP (2 letters, hyphen, 2 letters, hyphen, 2 letters, hyphen, 3 letters)
    patterns = [
        r'[A-Z]{2}-[A-Z]{2}-[A-Z]{2}-[A-Z]{3}',  # Standard: YC-ST-WA-EIP
        r'[A-Z]{2}-[A-Z]{2}-[A-Z]{2,4}',          # Shorter: YC-ST-WA
        r'[A-Z]{2,3}-\d{3,4}',                     # With numbers: YC-01-002
        r'[A-Z]{2,4}-\d{2,4}[A-Z]?',               # Mixed: YC-123A
        r'[A-Z]{2}-[A-Z]{2}-[A-Z]{3}',             # 3-part: YC-ST-WA
        r'[A-Z]{2}-[A-Z]{2}-\d{3}',                 # Numbered: YC-ST-001
    ]
    
    for pattern_regex in patterns:
        matches = re.findall(pattern_regex, query_upper)
        if matches:
            logger.debug(f"   Regex pattern match: {matches[0]}")
            return matches[0], 0.9
    
    # Strategy 3: Look for any element-like pattern (generic)
    general_regex = r'[A-Z]{2,}-[A-Z0-9]{2,}-[A-Z0-9]{2,}'
    matches = re.findall(general_regex, query_upper)
    if matches:
        logger.debug(f"   General pattern match: {matches[0]}")
        return matches[0], 0.7
    
    # Strategy 4: Look for element type + identifier pattern
    for etype in ELEMENT_TYPES.keys():
        if etype in query_lower:
            words = query_upper.split()
            for i, word in enumerate(words):
                if word.isalnum() and len(word) > 2:
                    if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                        logger.debug(f"   Type+ID pattern: {etype} {word}")
                        return word, 0.6
    
    return None, 0.0


def is_element_query(query: str) -> bool:
    """
    Check if query is asking about a specific element
    """
    if not query:
        return False
    
    query_lower = query.lower()
    
    for phrase in ELEMENT_QUERY_PHRASES:
        if phrase in query_lower:
            return True
    
    element, confidence = extract_element_from_query(query)
    return element is not None and confidence >= 0.6


def get_element_name_from_query(query: str) -> Optional[str]:
    """
    Get just the element name from query
    """
    element, _ = extract_element_from_query(query)
    return element


def get_all_element_patterns() -> List[str]:
    """Return all known element patterns"""
    return ELEMENT_PATTERNS.copy()


def get_element_type_from_query(query: str) -> Optional[str]:
    """
    Extract element type (wall, beam, etc.) from query
    """
    if not query:
        return None
    
    query_lower = query.lower()
    
    for etype, keywords in ELEMENT_TYPES.items():
        for keyword in keywords:
            if keyword in query_lower:
                return etype
    
    return None


# ==================== NEW FUNCTIONS (Fixes the Import Error) ====================

def extract_numbers(text: str) -> List[float]:
    """
    Extract numbers from text (useful for dimension queries)
    This function was missing - now added!
    
    Args:
        text: Input text
    
    Returns:
        list: List of found numbers
    """
    if not text:
        return []
    
    # Find all numbers (including decimals)
    number_pattern = r'\d+(?:\.\d+)?'
    matches = re.findall(number_pattern, text)
    return [float(m) for m in matches]


def contains_dimension_query(query: str) -> bool:
    """
    Check if query is about dimensions (height, width, thickness)
    This function was missing - now added!
    
    Args:
        query: Natural language query
    
    Returns:
        bool: True if dimension-related
    """
    if not query:
        return False
    
    query_lower = query.lower()
    dimension_keywords = [
        'height', 'width', 'thickness', 'length', 'depth',
        'tall', 'wide', 'thick', 'long', 'deep',
        'size', 'dimension', 'measurement', 'diameter',
        'area', 'volume', 'perimeter', 'span'
    ]
    
    for keyword in dimension_keywords:
        if keyword in query_lower:
            return True
    
    return False


# ==================== PRIORITY HANDLING ====================

def get_priority_boost(priority: str, query: str = None) -> float:
    """
    Get score boost based on priority
    """
    priority_lower = priority.lower() if priority else ''
    
    if 'high' in priority_lower:
        base_boost = 0.3
    elif 'medium' in priority_lower:
        base_boost = 0.15
    else:
        base_boost = 0.0
    
    if query:
        query_lower = query.lower()
        for level, keywords in PRIORITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if level == 'high':
                        return base_boost + 0.2
                    elif level == 'medium':
                        return base_boost + 0.1
    
    return base_boost


def format_priority_icon(priority: str) -> str:
    """
    Get icon for priority level
    """
    priority_lower = priority.lower() if priority else ''
    if 'high' in priority_lower:
        return "🔴"
    elif 'medium' in priority_lower:
        return "🟡"
    else:
        return "🟢"


# ==================== QUERY CLASSIFICATION ====================

def classify_query_intent(query: str) -> Dict:
    """
    Classify the intent of a query
    """
    if not query:
        return {'intent': 'unknown', 'score': 0.0}
    
    query_lower = query.lower()
    intents = {
        'compliance': ['non compliant', 'violation', 'fails', 'issue', 'problem', 'check'],
        'requirement': ['requirement', 'req', 'rule', 'spec', 'must', 'shall'],
        'product': ['product', 'material', 'manufacturer', 'cost', 'price'],
        'element': ['wall', 'beam', 'column', 'slab', 'element', 'find', 'show', 'list'],
        'regulation': ['code', 'regulation', 'nbc', 'standard'],
        'comparison': ['compare', 'vs', 'versus', 'difference', 'stricter', 'weaker']
    }
    
    scores = {}
    for intent, keywords in intents.items():
        score = 0
        for keyword in keywords:
            if keyword in query_lower:
                score += 1
        scores[intent] = score / len(keywords) if keywords else 0
    
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    return {
        'intent': best_intent if best_score > 0 else 'unknown',
        'score': best_score,
        'all_scores': scores
    }


def is_list_all_query(query: str) -> bool:
    """
    Check if query is asking to list all elements of a type
    """
    if not query:
        return False
    
    query_lower = query.lower()
    list_patterns = [
        r'show (?:me )?all',
        r'list all',
        r'find all',
        r'get all',
        r'display all'
    ]
    
    for pattern in list_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False


# ==================== TEST FUNCTION ====================

def test_element_utils():
    """Test all utility functions including new ones"""
    print("🧪 Testing element_utils.py")
    print("=" * 50)
    
    test_queries = [
        "Show issues for YC-ST-WA-EIP",
        "What's wrong with YC-AR-WA-IIP?",
        "Find column YC-ST-SC-CIP",
        "Show beam YC-ST-SF-BIP",
        "Walls with thickness > 200mm",
        "Elements taller than 3 meters",
        "Show me high priority violations",
        "List all walls in the model",
        "Check compliance of external walls"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        element, confidence = extract_element_from_query(query)
        print(f"   Element: {element} (confidence: {confidence:.2f})")
        
        is_element = is_element_query(query)
        print(f"   Is element query: {is_element}")
        
        etype = get_element_type_from_query(query)
        print(f"   Element type: {etype}")
        
        # Test new functions
        numbers = extract_numbers(query)
        print(f"   Numbers found: {numbers}")
        
        is_dimension = contains_dimension_query(query)
        print(f"   Is dimension query: {is_dimension}")
        
        intent = classify_query_intent(query)
        print(f"   Intent: {intent['intent']} (score: {intent['score']:.2f})")


def generate_more_test_queries() -> List[Dict]:
    """
    Generate comprehensive test queries for all layers
    """
    queries = []
    
    # L1 Element Queries
    l1_queries = [
        # By element type
        "Show me all walls in the model",
        "List all beams",
        "Display all columns",
        "Find all doors",
        "Show me all windows",
        "List all slabs",
        "Display all footings",
        "Find all openings",
        
        # By specific element ID
        "Show me element YC-ST-WA-EIP",
        "Find wall YC-AR-WA-IIP",
        "Locate column YC-ST-SC-CIP",
        "Display beam YC-ST-SF-BIP",
        "Show door YC-AR-DO-INT",
        "Find window YC-AR-WN-CAW",
        
        # By properties
        "Show elements with height > 3m",
        "Find walls with thickness 200mm",
        "List fire-rated doors",
        "Find all load-bearing walls",
    ]
    
    # L4 Regulation Queries
    l4_queries = [
        "What does NBC say about fire resistance?",
        "Show fire safety regulations",
        "What are the minimum wall thickness requirements?",
        "Building code for external walls",
        "Fire rating requirements for walls",
    ]
    
    # L5 Requirement Queries
    l5_queries = [
        "What are the project requirements for fire rating?",
        "Show all high priority requirements",
        "List requirements for external walls",
        "What is REQ-FIRE-001?",
        "Find requirements for door hardware",
    ]
    
    # L124 Compliance Queries
    l124_queries = [
        "Which walls violate fire code?",
        "Show me non-compliant elements",
        "Find walls that don't meet regulations",
        "What elements fail building code?",
        "Show fire code violations",
    ]
    
    # L125 Compliance Queries
    l125_queries = [
        "Which elements don't meet requirements?",
        "Show non-compliant external walls",
        "What walls fail fire rating?",
        "Find mismatches between model and rules",
        "List all compliance issues",
    ]
    
    # L45 Comparison Queries
    l45_queries = [
        "Compare requirements with regulations",
        "Is project stricter than code?",
        "Show differences between rules and codes",
        "Where does project exceed code?",
        "Are requirements weaker than regulations?",
    ]
    
    all_queries = (
        [(q, "l1") for q in l1_queries] +
        [(q, "l4") for q in l4_queries] +
        [(q, "l5") for q in l5_queries] +
        [(q, "l124") for q in l124_queries] +
        [(q, "l125") for q in l125_queries] +
        [(q, "l45") for q in l45_queries]
    )
    
    for i, (query_text, qtype) in enumerate(all_queries):
        queries.append({
            "id": f"test_{qtype}_{i:03d}",
            "query": query_text,
            "query_type": qtype,
            "expected_retriever": {
                "l1": "l1",
                "l4": "l4",
                "l5": "l5",
                "l124": "l1_l2_l4",
                "l125": "l1_l2_l5",
                "l45": "l4_l5"
            }[qtype],
            "description": f"Test {qtype} query"
        })
    
    return queries


if __name__ == "__main__":
    test_element_utils()
    
    # Generate test queries
    queries = generate_more_test_queries()
    print(f"\n✅ Generated {len(queries)} test queries")
    
    # Save to file
    import json
    from datetime import datetime
    
    output = {
        "generated": datetime.now().isoformat(),
        "total_queries": len(queries),
        "queries": queries
    }
    
    with open("test_queries_comprehensive.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"   Saved to test_queries_comprehensive.json")