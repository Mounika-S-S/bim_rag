"""
src/reasoning/structured_responder.py

Anti-hallucination layer that intercepts every user query and decides
whether the LLM is even needed.

Decision tree:
1. Parse entity type + property from query
2. Check FAISS metadata for existence
3. Branch:
   A) Property NOT in project   → deterministic "not found" response, list what EXISTS
   B) Compliance data exists     → format compliant/non-compliant response (LLM only for explanation)
   C) Property found, no compl.  → describe what we know, LLM explains
   D) General question           → normal RAG + LLM
"""
import re
import json
from typing import Optional
from src.rag.unified_vector_store import UnifiedVectorStore

# ── Entity type keywords ──────────────────────────────────────────────
ENTITY_KEYWORDS = {
    "wall": "wall", "walls": "wall",
    "beam": "beam", "beams": "beam",
    "column": "column", "columns": "column",
    "slab": "slab", "slabs": "slab",
    "roof": "roof",
    "stair": "stair", "staircase": "stair",
    "door": "door", "doors": "door",
    "window": "window",
    "foundation": "foundation", "footing": "foundation",
    "basement": "basement",
    "floor": "floor",
    "pile": "pile",
    "ramp": "ramp",
    "corridor": "corridor",
}

# ── Property keywords → canonical name ───────────────────────────────
PROPERTY_KEYWORDS = {
    "fire rating": "FireRating_min",
    "fire resistance": "FireRating_min",
    "fire": "FireRating_min",
    "thickness": "Thickness_mm",
    "height": "Height_mm",
    "width": "Width_mm",
    "depth": "Depth_mm",
    "length": "Length_mm",
    "compressive strength": "CompressiveStrength_MPa",
    "strength": "CompressiveStrength_MPa",
    "grade": "Grade",
    "concrete grade": "ConcreteGrade",
    "steel grade": "SteelGrade",
    "material": "Material",
    "manufacturer": "Manufacturer",
    "cover": "Cover_mm",
    "setback": "Setback",
    "fsi": "FSI",
    "parking": "Parking",
    "area": "Qty_Area",
    "volume": "Qty_Volume",
    "weight": "UnitWeight_kg",
    "load bearing": "IsLoadBearing",
    "external": "IsExternal",
}

COMPLIANCE_QUERIES = re.compile(
    r"\b(complian[ct]|comply|violat|non.complian[ct]|pass|fail|meet|conform)\b",
    re.IGNORECASE,
)
PROPERTY_QUERY = re.compile(
    r"\b(what is|what are|show|list|get|find|check|give|tell)\b",
    re.IGNORECASE,
)


def _detect_entity(query: str) -> Optional[str]:
    q = query.lower()
    for kw, norm in ENTITY_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", q):
            return norm
    return None


def _detect_property(query: str) -> Optional[str]:
    q = query.lower()
    # Longest match first
    for kw in sorted(PROPERTY_KEYWORDS, key=len, reverse=True):
        if kw in q:
            return PROPERTY_KEYWORDS[kw]
    return None


class StructuredResponder:
    """
    Intercepts queries and routes to deterministic or LLM-grounded response.
    Always requires the store to be loaded before calling respond().
    """

    def __init__(self, store: UnifiedVectorStore):
        self.store = store

    def respond(self, query: str, llm_client) -> str:
        """
        Main entry point. Returns a structured string response.
        llm_client: LLMClient instance for explanation generation only.
        """
        entity_type = _detect_entity(query)
        property_name = _detect_property(query)
        is_compliance_query = bool(COMPLIANCE_QUERIES.search(query))

        # ── Path A: User asks about specific element + property ───────
        if entity_type and property_name:
            return self._handle_specific(query, entity_type, property_name,
                                         is_compliance_query, llm_client)

        # ── Path B: Compliance-only query ─────────────────────────────
        if is_compliance_query:
            return self._handle_compliance_summary(query, entity_type, llm_client)

        # ── Path C: List query ("show all elements", "what elements exist") ─
        if entity_type and not property_name:
            return self._handle_entity_info(query, entity_type, llm_client)

        # ── Path D: General RAG ───────────────────────────────────────
        return self._handle_general(query, llm_client)

    # ─────────────────────────────────────────────────────────────────
    # Path A: Element + Property lookup
    # ─────────────────────────────────────────────────────────────────
    def _handle_specific(self, query: str, entity_type: str, property_name: str,
                         is_compliance: bool, llm) -> str:
        # 1. Check compliance records first for this element+property
        compliance_chunks = self.store.search_with_metadata(
            query=f"{entity_type} {property_name}",
            k=10,
            filter_element_type=entity_type,
        )
        # Look for a compliance chunk for this property
        comp_chunk = None
        for c in compliance_chunks:
            m = c["meta"]
            if (m.get("chunk_type") == "compliance"
                    and property_name.lower() in c["text"].lower()):
                comp_chunk = c
                break

        if comp_chunk:
            return self._format_compliance_response(comp_chunk, query, llm)

        # 2. Check if element+property exists in L1/L2 knowledge
        prop_chunks = self.store.search_with_metadata(
            query=f"{entity_type} {property_name}",
            k=10,
            filter_element_type=entity_type,
        )
        el_chunks = [c for c in prop_chunks if
                     c["meta"].get("layer") in ("L1", "L2")
                     and property_name.lower() in c["text"].lower()]

        if not el_chunks:
            # Property genuinely not in project
            available_types = self.store.get_all_element_types()
            available_props = self.store.get_all_properties_for_type(entity_type)
            return self._format_not_found(entity_type, property_name,
                                          available_types, available_props)

        # Property found but no compliance check done yet
        values_text = "\n".join(c["text"] for c in el_chunks[:3])
        context = f"Project data for {entity_type} {property_name}:\n{values_text}"
        explanation = llm.reason(query, context)
        return (
            f"📋 **{property_name} for {entity_type.title()} elements**\n\n"
            f"{values_text}\n\n"
            f"💡 **Analysis:**\n{explanation}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Path B: Compliance summary
    # ─────────────────────────────────────────────────────────────────
    def _handle_compliance_summary(self, query: str, entity_type: Optional[str], llm) -> str:
        filter_et = entity_type  # may be None for global scan
        chunks = self.store.search_with_metadata(query, k=15,
                                                  filter_element_type=filter_et)
        non_compliant = [c for c in chunks if c["meta"].get("status") == "NON_COMPLIANT"]
        missing = [c for c in chunks if c["meta"].get("status") == "MISSING_PROPERTY"]
        compliant = [c for c in chunks if c["meta"].get("status") == "COMPLIANT"]

        if not non_compliant and not missing:
            if compliant:
                names = set(c["meta"].get("element_name", "") for c in compliant)
                return (
                    f"✅ **All checked {entity_type or 'element'} elements are COMPLIANT.**\n\n"
                    f"Compliant elements: {', '.join(n for n in names if n)}\n\n"
                    + "\n".join(c["text"] for c in compliant[:3])
                )
            # No compliance data at all
            return ("⚠️ No compliance check results found. "
                    "Please run inference (option 7) before querying compliance.")

        lines = []
        if non_compliant:
            lines.append(f"❌ **NON-COMPLIANT ({len(non_compliant)} issues found):**")
            for c in non_compliant[:5]:
                m = c["meta"]
                lines.append(f"  • {m.get('element_name','?')} — {m.get('property','?')}: {c['text'][:200]}")

        if missing:
            lines.append(f"\n⚠️ **MISSING PROPERTIES ({len(missing)} elements):**")
            for c in missing[:3]:
                m = c["meta"]
                lines.append(f"  • {m.get('element_name','?')}: {c['text'][:150]}")

        context = "\n".join(c["text"] for c in (non_compliant + missing)[:5])
        suggestion = llm.reason(
            f"Based on these compliance issues, what are the key corrective actions?\n{query}",
            context,
        )
        lines.append(f"\n💡 **Recommended Actions:**\n{suggestion}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    # Path C: Entity info without specific property
    # ─────────────────────────────────────────────────────────────────
    def _handle_entity_info(self, query: str, entity_type: str, llm) -> str:
        chunks = self.store.search_with_metadata(query, k=8, filter_element_type=entity_type)
        if not chunks:
            available = self.store.get_all_element_types()
            return (
                f"⚠️ No data found for **{entity_type}** elements in this project.\n\n"
                f"Available element types: {', '.join(available) if available else 'None yet — upload L1/L2 data first.'}"
            )
        props = self.store.get_all_properties_for_type(entity_type)
        context = "\n".join(c["text"] for c in chunks[:5])
        explanation = llm.reason(query, context)
        return (
            f"📦 **{entity_type.title()} Elements in Project**\n\n"
            f"Known properties: {', '.join(props[:15]) if props else 'None extracted yet'}\n\n"
            f"{context[:800]}\n\n"
            f"💡 **Summary:**\n{explanation}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Path D: General semantic RAG
    # ─────────────────────────────────────────────────────────────────
    def _handle_general(self, query: str, llm) -> str:
        chunks = self.store.search(query, k=4)
        context = "\n".join(chunks)
        answer = llm.reason(query, context)
        return f"💬 **Answer:**\n{answer}"

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────
    def _format_compliance_response(self, comp_chunk: dict, query: str, llm) -> str:
        m = comp_chunk["meta"]
        status = m.get("status", "UNKNOWN")
        ename = m.get("element_name", "element")
        etype = m.get("element_type", "element")
        prop = m.get("property", "property")
        chunk_text = comp_chunk["text"]

        if status == "COMPLIANT":
            explanation = llm.reason(
                f"Explain concisely why this {etype} is compliant for {prop}. Query: {query}",
                chunk_text,
            )
            return (
                f"✅ **COMPLIANT** — {etype.title()} '{ename}'\n\n"
                f"**Property:** {prop}\n"
                f"**Details:** {chunk_text[:400]}\n\n"
                f"💡 **Reasoning:**\n{explanation}"
            )
        elif status == "NON_COMPLIANT":
            explanation = llm.reason(
                f"Explain why this {etype} fails {prop} compliance and what action is needed. Query: {query}",
                chunk_text,
            )
            return (
                f"❌ **NON-COMPLIANT** — {etype.title()} '{ename}'\n\n"
                f"**Property:** {prop}\n"
                f"**Details:** {chunk_text[:400]}\n\n"
                f"💡 **Reasoning & Suggestion:**\n{explanation}"
            )
        elif status == "MISSING_PROPERTY":
            avail_props = self.store.get_all_properties_for_type(etype)
            return (
                f"⚠️ **PROPERTY NOT FOUND** — {etype.title()} '{ename}'\n\n"
                f"**Requested property:** `{prop}` is **not present** in L1 (IFC) or L2 (product) data.\n\n"
                f"**Available properties for {etype}:** {', '.join(avail_props[:20]) if avail_props else 'None yet'}\n\n"
                f"**Action:** Add `{prop}` to the product data sheet (L2) for this element type."
            )
        return f"ℹ️ {chunk_text}"

    @staticmethod
    def _format_not_found(entity_type: str, property_name: str,
                          available_types: list, available_props: list) -> str:
        return (
            f"⚠️ **Property Not Found in Project**\n\n"
            f"The property **`{property_name}`** was **not found** for **`{entity_type}`** elements "
            f"in the current project's L1 (IFC) or L2 (product) data.\n\n"
            f"**Available element types in this project:**\n"
            + ("  " + "\n  ".join(f"• {t}" for t in available_types) if available_types
               else "  None yet — please upload and process L1/L2 files first.")
            + f"\n\n**Properties available for `{entity_type}` elements:**\n"
            + ("  " + "\n  ".join(f"• {p}" for p in available_props[:20]) if available_props
               else f"  No `{entity_type}` elements found in the project yet.")
        )
