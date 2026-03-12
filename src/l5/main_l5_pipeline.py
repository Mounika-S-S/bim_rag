# src/l5/main_l5_pipeline.py
"""
Unified L5 Pipeline - Handles both Excel and PDF requirements
Uses unified RequirementParser for all file types
"""
import os
from src.ingestion.requirement_parser import RequirementParser
from src.core.json_storage import JSONStorage


class L5Pipeline:
    """
    Unified pipeline for Layer 5 requirements
    
    Features:
    - Auto-detects file type from extension
    - Excel: Uses RequirementParser with intelligent header detection & noise removal
    - PDF: Uses RequirementParser with table extraction & pattern matching
    - Returns standardized requirement records
    """
    
    def __init__(self):
        # Use the unified parser for both Excel and PDF
        self.parser = RequirementParser()
    
    def parse(self, file_path, file_type=None, sheet_name=None):
        """
        Parse requirements from file using unified parser
        
        Args:
            file_path: Path to Excel or PDF file
            file_type: 'excel' or 'pdf' (auto-detected if None)
            sheet_name: Sheet name for Excel files (ignored for PDF)
        
        Returns:
            List of requirement records in standardized format
        """
        # Auto-detect file type if not specified
        if file_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.xlsx', '.xls', '.xlsm']:
                file_type = 'excel'
            else:
                file_type = 'pdf'
        
        print(f"\n📄 Parsing {file_type.upper()} requirements from: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # Use unified parser
        requirements = self.parser.parse(
            file_path=file_path,
            file_type=file_type,
            sheet_name=sheet_name
        )
        
        if not requirements:
            print("❌ No requirements extracted.")
            return []
        
        print(f"✅ Extracted {len(requirements)} requirements")
        
        # Show sample if available
        if requirements:
            print("\n📊 Sample Requirement:")
            sample = requirements[0]
            props = sample.get('properties', {})
            print(f"   ID: {props.get('RequirementID', 'N/A')}")
            print(f"   Element: {props.get('ElementType', 'N/A')}")
            print(f"   Rule: {props.get('Property', 'N/A')} {props.get('Operator', 'N/A')} {props.get('RequiredValue', 'N/A')} {props.get('Unit', 'N/A')}")
            print(f"   Priority: {props.get('Priority', 'N/A')}")
        
        return requirements
    
    def parse_batch(self, file_paths, sheet_name=None):
        """
        Parse multiple files and combine results
        
        Args:
            file_paths: List of file paths (Excel or PDF)
            sheet_name: Sheet name for Excel files
        
        Returns:
            Combined list of requirement records
        """
        all_requirements = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                reqs = self.parse(file_path, sheet_name=sheet_name)
                all_requirements.extend(reqs)
            else:
                print(f"⚠️ File not found: {file_path}")
        
        return all_requirements