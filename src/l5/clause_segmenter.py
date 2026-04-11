"""
L5 Clause Segmenter — handles both PDF (table + text) and Excel.
Company-specific project requirements.
"""
import os
import re
import pandas as pd
import pdfplumber
from src.ingestion.base_pdf_cleaner import BasePDFCleaner

cleaner = BasePDFCleaner()

RELEVANT_KEYWORDS = [
    "shall", "must", "required", "minimum", "maximum", "provide",
    "install", "supply", "submit", "comply", "ensure", "maintain",
    "complete", "deliver", "schedule", "rate", "quantity", "unit",
    "specification", "standard", "code", "drawing", "approval",
]


def _is_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in RELEVANT_KEYWORDS) and len(text) > 20


class ClauseSegmenter:

    def segment(self, file_path: str) -> list[dict]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            return self._from_excel(file_path)
        else:
            return self._from_pdf(file_path)

    # ------------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------------
    def _from_excel(self, path: str) -> list[dict]:
        rows = []
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            print(f"L5 Excel error: {e}")
            return []

        for sheet in xl.sheet_names:
            df = xl.parse(sheet, dtype=str).fillna("")
            for _, row in df.iterrows():
                vals = [str(v).strip() for v in row.values if str(v).strip()]
                if not vals:
                    continue
                # Try standard columns
                code = str(row.get("Code", row.get("Item Code", row.get("item_code", "")))).strip()
                desc = str(row.get("Description", row.get("description", ""))).strip()
                unit = str(row.get("Unit", row.get("unit", ""))).strip()
                rate = str(row.get("Rate", row.get("rate", ""))).strip()

                if not desc:
                    desc = " | ".join(vals)

                if _is_relevant(desc) or code:
                    rows.append({
                        "code": code or f"L5_{len(rows)}",
                        "description": desc,
                        "unit": unit,
                        "rate": rate,
                        "source_document": os.path.basename(path),
                    })
        return rows

    # ------------------------------------------------------------------
    # PDF — table-first, then text
    # ------------------------------------------------------------------
    def _from_pdf(self, path: str) -> list[dict]:
        rows = []
        doc_name = os.path.basename(path)

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                h, w = page.height, page.width
                try:
                    cropped = page.within_bbox((0, 0.07 * h, w, 0.93 * h))
                except Exception:
                    cropped = page

                text_raw = cropped.extract_text(x_tolerance=2, y_tolerance=2) or ""
                if cleaner._is_noise_page(text_raw):
                    continue

                # Try tables
                tables = cropped.extract_tables()
                extracted_from_table = False
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    headers = [str(c or "").strip().lower() for c in table[0]]
                    for data_row in table[1:]:
                        if not data_row:
                            continue
                        row_map = {headers[i]: str(data_row[i] or "").strip()
                                   for i in range(min(len(headers), len(data_row)))}
                        code = row_map.get("code", row_map.get("item", row_map.get("no.", "")))
                        desc = row_map.get("description", row_map.get("item description",
                                row_map.get("specification", "")))
                        unit = row_map.get("unit", row_map.get("uom", ""))
                        rate = row_map.get("rate", row_map.get("amount", ""))

                        if not desc:
                            desc = " | ".join(v for v in row_map.values() if v)

                        if desc and _is_relevant(desc):
                            rows.append({
                                "code": code or f"L5_{page_num}_{len(rows)}",
                                "description": desc,
                                "unit": unit,
                                "rate": rate,
                                "source_document": doc_name,
                            })
                            extracted_from_table = True

                # Fallback to text paragraphs
                if not extracted_from_table:
                    text = cleaner._clean_page(text_raw)
                    for para in re.split(r"\n{2,}", text):
                        para = para.strip()
                        if _is_relevant(para):
                            rows.append({
                                "code": f"L5_{page_num}_{len(rows)}",
                                "description": para,
                                "unit": "",
                                "rate": "",
                                "source_document": doc_name,
                            })

        return rows