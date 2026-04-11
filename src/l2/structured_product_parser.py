# src/l2/structured_product_parser.py
"""
Structured product parser for L2.
- PDF: extracts tables via pdfplumber → key-value property rows
- Excel: reads pandas DataFrame columns directly
Outputs canonical property names for cross-layer alignment with L1 IFC elements.
"""
import os
import re
import pandas as pd
import pdfplumber
from src.ingestion.base_pdf_cleaner import BasePDFCleaner

# Canonical property name aliases (any PDF header variation → canonical key)
PROPERTY_ALIASES = {
    # Fire
    "fire rating": "FireRating_min",
    "fire resistance": "FireRating_min",
    "fire rating (min)": "FireRating_min",
    "fire resistance period": "FireRating_min",
    # Thickness
    "thickness": "Thickness_mm",
    "thickness (mm)": "Thickness_mm",
    "overall thickness": "Thickness_mm",
    "wall thickness": "Thickness_mm",
    # Strength
    "compressive strength": "CompressiveStrength_MPa",
    "compressive strength (mpa)": "CompressiveStrength_MPa",
    "fck": "CompressiveStrength_MPa",
    "characteristic strength": "CompressiveStrength_MPa",
    # Grade / material
    "concrete grade": "ConcreteGrade",
    "steel grade": "SteelGrade",
    "grade": "Grade",
    "material": "Material",
    # Dimensions
    "length": "Length_mm",
    "length (mm)": "Length_mm",
    "width": "Width_mm",
    "width (mm)": "Width_mm",
    "height": "Height_mm",
    "height (mm)": "Height_mm",
    "depth": "Depth_mm",
    "depth (mm)": "Depth_mm",
    # Misc
    "manufacturer": "Manufacturer",
    "brand": "Manufacturer",
    "product name": "ProductName",
    "item": "ProductName",
    "element type": "ElementType",
    "type": "ElementType",
    "unit weight": "UnitWeight_kg",
    "density": "Density_kg_m3",
    "rebar diameter": "RebarDiameter_mm",
    "cover": "Cover_mm",
    "cover (mm)": "Cover_mm",
}

# Element type keywords → normalized
ELEMENT_KEYWORDS = {
    "wall": "wall", "walls": "wall",
    "beam": "beam", "beams": "beam",
    "column": "column", "columns": "column",
    "slab": "slab", "slabs": "slab",
    "roof": "roof",
    "stair": "stair", "staircase": "stair",
    "door": "door",
    "window": "window",
    "foundation": "foundation", "footing": "foundation",
    "pile": "pile",
}

NUMBER_RE = re.compile(r"[\d]+(?:\.\d+)?")


def _canonical(raw_key: str) -> str:
    """Resolve a raw header string to a canonical property name."""
    clean = raw_key.strip().lower()
    return PROPERTY_ALIASES.get(clean, raw_key.strip())


def _extract_element_type(text: str) -> str | None:
    t = text.lower()
    for kw, norm in ELEMENT_KEYWORDS.items():
        if kw in t:
            return norm
    return None


def _parse_value(raw: str):
    """Try to extract numeric value from string like '120 min' or '250mm'."""
    if raw is None:
        return raw
    raw = str(raw).strip()
    m = NUMBER_RE.search(raw)
    if m and len(raw) < 30:  # short enough to be a value cell
        val = float(m.group())
        return int(val) if val == int(val) else val
    return raw if raw else None


class StructuredProductParser:
    """Parse L2 product data from PDF (tables) or Excel into canonical records."""

    def __init__(self):
        self.cleaner = BasePDFCleaner()

    # ------------------------------------------------------------------
    # PDF parsing — table-first, then key-value paragraph fallback
    # ------------------------------------------------------------------
    def parse_pdf(self, pdf_path: str) -> list[dict]:
        records = []
        doc_name = os.path.basename(pdf_path)

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # -- bounding box crop for headers/footers --
                h, w = page.height, page.width
                try:
                    cropped = page.within_bbox((0, 0.07 * h, w, 0.93 * h))
                except Exception:
                    cropped = page

                # -- try table extraction first --
                tables = cropped.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    table_records = self._parse_table(table, doc_name, page_num)
                    records.extend(table_records)

                # -- if no tables found on page, do paragraph key-value scan --
                if not tables:
                    text = cropped.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    if self.cleaner._is_noise_page(text):
                        continue
                    kv_records = self._parse_kv_text(text, doc_name, page_num)
                    records.extend(kv_records)

        return records

    def _parse_table(self, table: list, doc_name: str, page_num: int) -> list[dict]:
        """Convert a pdfplumber table to a list of canonical property dicts."""
        records = []
        headers = [str(c).strip() if c else "" for c in table[0]]

        # Detect if first row is actually headers
        if any(h.replace(" ", "").isdigit() for h in headers):
            return []  # looks like data not headers

        for row in table[1:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue

            props = {}
            element_type = None
            for header, cell in zip(headers, row):
                if not header:
                    continue
                canonical = _canonical(header)
                value = _parse_value(cell)
                if value is not None and str(value).strip() != "":
                    props[canonical] = value
                # detect element type from value or header
                if element_type is None:
                    et = _extract_element_type(str(cell or ""))
                    if et:
                        element_type = et
                if element_type is None:
                    element_type = _extract_element_type(header)

            if props:
                props["source_document"] = doc_name
                props["page_number"] = page_num
                records.append({
                    "properties": props,
                    "element_type_normalized": element_type,
                })

        return records

    def _parse_kv_text(self, text: str, doc_name: str, page_num: int) -> list[dict]:
        """Parse 'Key: Value' lines from unstructured product text."""
        props = {}
        element_type = None
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) > 200:
                continue
            # Match "Key : Value" or "Key - Value"
            m = re.match(r"^([A-Za-z][A-Za-z\s\(\)/]{2,40})\s*[:–-]\s*(.+)$", line)
            if m:
                key_raw, val_raw = m.group(1).strip(), m.group(2).strip()
                canonical = _canonical(key_raw)
                value = _parse_value(val_raw)
                if value is not None:
                    props[canonical] = value
                if element_type is None:
                    element_type = _extract_element_type(key_raw) or _extract_element_type(val_raw)

        if props:
            props["source_document"] = doc_name
            props["page_number"] = page_num
            return [{"properties": props, "element_type_normalized": element_type}]
        return []

    # ------------------------------------------------------------------
    # Excel parsing
    # ------------------------------------------------------------------
    def parse_excel(self, excel_path: str) -> list[dict]:
        records = []
        doc_name = os.path.basename(excel_path)

        try:
            xl = pd.ExcelFile(excel_path)
        except Exception as e:
            print(f"Error opening Excel {excel_path}: {e}")
            return []

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name, header=0, dtype=str)
            df = df.dropna(how="all")
            if df.empty:
                continue

            # Rename columns to canonical names
            col_map = {col: _canonical(col) for col in df.columns}
            df = df.rename(columns=col_map)

            for _, row in df.iterrows():
                props = {}
                element_type = None
                for col in df.columns:
                    val = _parse_value(row.get(col))
                    if val is not None and str(val).strip() not in ("", "nan", "None"):
                        props[col] = val
                    if element_type is None and col == "ElementType":
                        element_type = _extract_element_type(str(val or ""))

                if props:
                    props["source_document"] = doc_name
                    props["sheet"] = sheet_name
                    records.append({
                        "properties": props,
                        "element_type_normalized": element_type,
                    })

        return records
