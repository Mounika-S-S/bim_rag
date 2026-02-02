**BIM Compliance Intelligence System (RAG-Enabled)**
**Overview**

This project implements a BIM compliance intelligence system that analyzes IFC models against project requirements, product data, and regulations to identify non-compliant building elements and explain the results using Retrieval-Augmented Generation (RAG).

The system is designed with a clear separation between deterministic engineering logic and AI-based explanation:

✅ Compliance decisions are rule-based and auditable

✅ RAG retrieves verified evidence

✅ LLMs (optional) are used only for explanation, never for decision-making
**
What Problem Does This Solve?**

In real BIM workflows, information is fragmented:

Geometry → IFC files

Requirements → Excel sheets

Products → Manufacturer catalogs

Regulations → PDFs

Explanations → Manual, error-prone

This system unifies all these sources into a single, explainable compliance pipeline.

**High-Level Architecture**
User Question
   ↓
Query Router (intent detection)
   ↓
Vector DB (ChromaDB) — RAG retrieval
   ↓
Deterministic Results (mismatches, rules, products, regulations)
   ↓
( Optional ) LLM — explanation only
   ↓
Engineer-readable answer

**
Data Layers**
**Layer 1 — IFC Model (What is Built)**

Input: .ifc file

Output: ifc_walls.json

Extracts:

Wall IDs

Names

Types

Property sets (Psets)

**Layer 2 — Product Data (What It Is)**

Input: Manufacturer Excel / PDF

Output: products.json

Contains:

System type

Fire rating

Manufacturer

Constraints

**Layer 3 — Documents (Proof)**

Input: Technical PDFs

Output: documents.json

Used as:

Supporting evidence

RAG context only

**Layer 4 — Regulations (Authority)**

Input: Fire code PDFs

Output: regulations.json

Used to:

Ground explanations

Reference standards and codes

**Layer 5 — Project Requirements (Context)**

Input: Project Excel

Output: rules.json

Defines:

Required fire ratings

Scope (internal / external)

Priority and description

Compliance Engine (Core Logic)

The compliance engine compares:

IFC elements (Layer 1)

Against project requirements (Layer 5)

Using available product systems (Layer 2)

Output
mismatches.json


Example:

{
  "wall_id": "2DedXznHnDaeAWsrTB_qBp",
  "wall_name": "Basic Wall:Yttervägg Paroc",
  "issue": "No compliant product found",
  "required_fire_rating": "EI120",
  "wall_fire_rating": null
}


✔ Deterministic
✔ Explainable
✔ Auditable

**RAG Pipeline**
Vector Database

Engine: ChromaDB

Storage: Local (data/vector_db)

No cloud, no accounts, no API keys

Embedded Sources

mismatches.json

rules.json

products.json

documents.json

regulations.json

Query Routing (General Explanation RAG)

The system supports different question types using query routing:

Question Type	Example	Retrieved Sources
Compliance	“Which walls violate fire safety?”	mismatches
Project overview	“Explain the project”	documents, rules
Regulations	“Which regulation requires EI120?”	regulations
Products	“Which products are approved?”	products
Unsupported	“Total cost?”	Graceful fallback

This ensures relevant and precise retrieval.

Query Engine

Engineers can ask questions such as:

Which walls violate fire safety?

Why is wall X non-compliant?

What fire rating is required?

Which regulations apply?

Which products are approved?

The system retrieves verified context and presents it clearly.
**project structure**
<img width="495" height="870" alt="image" src="https://github.com/user-attachments/assets/30f08bb5-3a8f-4055-8fde-c8891110891f" />

**Technology Stack**

Python

ifcopenshell

pandas

PyPDF2

ChromaDB

sentence-transformers

(Optional) Ollama / Local LLM

**Design Principles**

Deterministic before generative

Explainability by design

Clear separation of logic and language

Privacy-first (local execution)

Engineering-grade correctness

**Current Status**

✅ IFC parsing
✅ Product ingestion
✅ Regulation ingestion
✅ Project requirements modeling
✅ Compliance detection
✅ RAG with query routing
✅ Engineer-readable answers
⏳ LLM explanation layer (optional)

**Intended Users**

BIM engineers

Fire safety reviewers

Compliance teams

Digital construction workflows
