# BIM Compliance Intelligence System

> **RAG-Enabled Compliance Analysis for Building Information Models**

A deterministic compliance intelligence system that parses IFC models, product data sheets, building regulations, and project requirements — then uses Retrieval-Augmented Generation (RAG) to answer compliance questions with verifiable, auditable reasoning.

---

## 🏗️ What Problem Does This Solve?

In real BIM workflows, compliance information lives in disconnected silos:

| Source | Format | Used For |
|---|---|---|
| Geometry | `.ifc` files | What is actually built |
| Product specs | Excel / PDF | Material properties & fire ratings |
| Process docs | PDF | Construction procedures |
| Regulations | PDF (Building codes) | Legal compliance thresholds |
| Requirements | Excel | Project-specific standards |

This system **unifies all five layers** into a single, explainable pipeline — enabling engineers to ask plain-language questions and receive answers grounded in deterministic compliance logic, not LLM hallucination.

---

## ✨ Key Design Principles

- **Deterministic before generative** — compliance verdicts are rule-based and auditable; LLMs are only used for *explanation*
- **Explainability by design** — every answer cites a source layer (L1–L5)
- **Privacy-first** — runs fully locally; no data leaves your machine
- **Engineering-grade correctness** — built for BIM engineers, not chatbot users

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│  L1 IFC   │  L2 Product  │  L3 Process  │  L4 Regs  │  L5 Req  │
└────────────────────────┬────────────────────────────────┘
                         │ Parsed → JSON records
                         ▼
┌─────────────────────────────────────────────────────────┐
│              COMPLIANCE ENGINE (Deterministic)          │
│   Matches L1 elements → L4 regulations → Flags issues  │
│   Output: compliance_inference.json                     │
└────────────────────────┬────────────────────────────────┘
                         │ All records + compliance results
                         ▼
┌─────────────────────────────────────────────────────────┐
│             UNIFIED VECTOR STORE (FAISS)                │
│   Chunked text embeddings for semantic retrieval         │
└────────────────────────┬────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │     User Question        │
            ▼                         ▼
┌───────────────────┐      ┌──────────────────────┐
│   Query Router     │      │  Structured Responder │
│  (Intent detect)  │      │  (Anti-hallucination)  │
└─────────┬─────────┘      └──────────┬─────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│              LLM LAYER (Groq / Gemini — Optional)       │
│   Used ONLY for explanation; never for decisions        │
└────────────────────────┬────────────────────────────────┘
                         ▼
              Engineer-readable answer
```

---

## 📂 Data Layers (L1–L5)

### L1 — IFC Model *(What is Built)*
- **Input:** `.ifc` file
- **Output:** `L1_ifc.json`
- Extracts element IDs, entity types, materials, geometric properties (Width, Length, FireRating, etc.)

### L2 — Product Data *(What It Is)*
- **Input:** Manufacturer Excel or PDF
- **Output:** `L2_product.json`
- Contains system type, fire ratings, manufacturer constraints, compressive strength

### L3 — Process Documents *(How It is Built)*
- **Input:** Technical PDFs, Excel
- **Output:** `L3_process.json`
- Construction procedures used as supporting RAG context

### L4 — Regulations *(The Authority)*
- **Input:** Building code PDFs (e.g., TN Combined Development & Building Rules 2019)
- **Output:** `L4_regulation.json`
- Structured clauses with numeric thresholds, element types, operators (≥, ≤, >)

### L5 — Project Requirements *(The Context)*
- **Input:** Project Excel or PDF
- **Output:** `L5_requirement.json`
- Defines required fire ratings, scope, compliance priority

---

## ⚙️ Compliance Engine

The deterministic engine (`src/inference/compliance_engine.py`) performs:

1. **Element matching** — L1 elements matched against L4 rules by `element_type_normalized`
2. **Property merging** — L2 product data fills gaps missing in L1 records
3. **Numeric evaluation** — thresholds checked with operators (`>=`, `<=`, `>`, `<`)
4. **Status assignment** — each element assigned `COMPLIANT`, `NON_COMPLIANT`, or `MISSING_PROPERTY`

**Output format** (`compliance_inference.json`):
```json
{
  "element_id": "1IwQbMyrr3zPxzRC_0XKuh",
  "element_name": "YC-ST-WA-EIP",
  "element_type": "IfcWall",
  "property": "FireRating_min",
  "effective_value": 60,
  "required_value": 120,
  "operator": ">=",
  "unit": "min",
  "status": "NON_COMPLIANT",
  "gap": -60,
  "suggestion": "Upgrade to a product with higher fire resistance rating meeting the required 120 min.",
  "source_rule": "Fire resistance clause 45...",
  "source_layer": "L4"
}
```

---

## 🔍 RAG Pipeline

| Component | Technology |
|---|---|
| Vector Store | FAISS (local, no cloud) |
| Embeddings | `sentence-transformers` |
| LLM (optional) | Groq API (Llama 3) |
| Query Routing | Custom intent detection |
| Anti-hallucination | `StructuredResponder` layer |

### Query Routing

| Question Type | Example | Retrieved Sources |
|---|---|---|
| Compliance | *"Which walls fail fire safety?"* | `compliance_inference.json` |
| Element info | *"What is the width of beams?"* | L1 + L2 chunks |
| Regulations | *"What does clause 58 say?"* | L4 regulation chunks |
| Product | *"Which products meet EI120?"* | L2 product chunks |
| General | *"Explain TNRB 2019 setbacks"* | Full semantic RAG |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- A Groq API key (free tier) or Gemini API key

### 1. Clone & install
```bash
git clone https://github.com/Mounika-S-S/bim_rag.git
cd bim_rag
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY or GEMINI_API_KEY
```

### 3. Run the CLI
```bash
python -m src.app
```

### 4. Run the API server
```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Run the frontend
```bash
cd frontend
# Open index.html in a browser or serve with live-server
```

---

## 🖥️ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/ping` | Health check |
| `GET` | `/projects` | List all projects |
| `POST` | `/projects` | Create a new project |
| `POST` | `/upload` | Upload and parse a layer file (L1–L5) |
| `POST` | `/run-inference` | Run compliance engine |
| `POST` | `/build-vector-store` | Embed all records into FAISS |
| `POST` | `/query` | Ask a compliance question |
| `POST` | `/feedback` | Submit rating on an answer |
| `POST` | `/history/clear` | Clear chat session history |
| `DELETE` | `/project` | Delete a project and all its data |

---

## 📁 Project Structure

```
bim_rag/
├── src/
│   ├── app.py                  # CLI entry point & orchestration
│   ├── api.py                  # FastAPI REST layer
│   ├── core/                   # Schema, JSON storage, model manager
│   ├── ingestion/              # IFC parser, L4 pipeline
│   ├── l2/                     # Product data pipeline
│   ├── l3/                     # Process document pipeline
│   ├── l4/                     # Regulation clause segmenter
│   ├── l5/                     # Requirements pipeline
│   ├── inference/              # Deterministic compliance engine
│   ├── rag/                    # Unified FAISS vector store
│   ├── reasoning/              # LLM client, structured responder
│   ├── retrieval/              # Query router, retriever
│   ├── embedding/              # Embedding utilities
│   ├── evaluation/             # RAGAS, BLEU, ROUGE metrics
│   ├── finetuning/             # DB logger for feedback loop
│   └── utils/                  # Shared helpers
├── data/
│   ├── processed/              # Per-project JSON records & FAISS index
│   └── chat_history/           # Session chat logs
├── frontend/                   # Web UI
├── evaluation/                 # Evaluation scripts & reports
├── notebooks/                  # Colab fine-tuning notebooks
├── tests/                      # Unit & integration tests
├── requirements.txt
└── .env
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| IFC Parsing | `ifcopenshell` |
| PDF Extraction | `pdfplumber`, `PyMuPDF`, `PyPDF2` |
| Excel Parsing | `pandas`, `openpyxl` |
| Vector Store | `faiss-cpu` |
| Embeddings | `sentence-transformers` |
| LLM API | `groq` (Llama 3) |
| REST API | `fastapi`, `uvicorn` |
| Evaluation | `ragas`, `rouge-score`, `nltk` |
| Fuzzy Matching | `fuzzywuzzy` |

---

## ✅ Current Status

| Feature | Status |
|---|---|
| IFC parsing (L1) | ✅ Complete |
| Product ingestion (L2) | ✅ Complete |
| Process ingestion (L3) | ✅ Complete |
| Regulation parsing (L4) | ✅ Complete |
| Requirements modeling (L5) | ✅ Complete |
| Deterministic compliance engine | ✅ Complete |
| FAISS vector store | ✅ Complete |
| Structured anti-hallucination responder | ✅ Complete |
| FastAPI REST layer | ✅ Complete |
| Web frontend | ✅ Complete |
| Chat history & sessions | ✅ Complete |
| Feedback logging for fine-tuning | ✅ Complete |
| RAGAS / BLEU / ROUGE evaluation | ✅ Complete |
| LLM fine-tuning pipeline (Colab) | ⏳ In Progress |

---

## 👥 Intended Users

- **BIM Engineers** verifying model compliance before submission
- **Fire Safety Reviewers** checking against building codes
- **Compliance Teams** conducting regulatory audits
- **Digital Construction Teams** integrating AI reasoning into BIM workflows

---

## 📄 License

This project is for academic and research purposes. See `LICENSE` for details.
