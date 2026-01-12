# bim-rag
#folder structure
<img width="684" height="735" alt="image" src="https://github.com/user-attachments/assets/31d81e74-a72e-4202-a476-2203760de06d" />

└─ README.md
Transforming Product Data Digitally
Yatzar Asset began when we recognised a common issue across the industry. Product information was scattered across PDFs, low quality images, outdated brochures and incomplete specifications. This created delays, confusion and rework in BIM driven workflows.

We wanted to create a platform where manufacturers could publish complete digital representations of their products and where professionals could easily discover and download building assets they can trust. Over time, as the use of Revit, IFC, ArchiCAD, SketchUp and digital twin technology expanded, our work evolved to meet global expectations for quality and accuracy

Layer 1: IFC model (what is built)
Layer 2: Product data (what it actually is)
Layer 3: Documents (proof)
Layer 4: Regulations (rules)
Layer 5: Project requirements (context)

Overview

This project implements a BIM-aware compliance intelligence system that analyzes IFC models against project requirements, product data, and regulations to identify mismatches and explain them using Retrieval-Augmented Generation (RAG).

The system is designed for engineering correctness, auditability, and trust.
All compliance decisions are made deterministically using rule logic.
Large Language Models (LLMs) are used only for explanation, not decision-making.

Key Capabilities

Parse IFC models (.ifc) to extract BIM elements

Ingest manufacturer product data (Excel / PDF)

Ingest regulatory documents (PDF)

Model project-specific requirements (Excel)

Perform deterministic compliance checks

Detect and store fire-safety mismatches

Enable natural-language querying using RAG

Generate clear, engineer-readable explanations

System Architecture:
User Query
   ↓
Rule Engine (deterministic)
   ↓
Structured Results (mismatches, rules, products)
   ↓
Vector Database (ChromaDB)
   ↓
RAG Retrieval (relevant evidence)
   ↓
LLM (explanation only)
   ↓
Final Answer


Data Layers
Layer 1 — IFC Model (What is Built)

Input: IFC file

Output: ifc_walls.json

Extracted data:

Element IDs

Names

Types

Property sets (Psets)

Layer 2 — Product Data (What It Is)

Input: Manufacturer Excel / PDF

Output: products.json

Contains:

System type

Fire rating

Constraints

Approved configurations

Layer 3 — Documents (Proof)

Input: Technical PDFs

Output: documents.json

Purpose:

Evidence for RAG

Traceability

Layer 4 — Regulations (Authority)

Input: Fire code PDFs

Output: regulations.json

Purpose:

Regulatory grounding

Reference during explanation

Layer 5 — Project Requirements (Context)

Input: Project Excel

Output: rules.json

Defines:

Required fire ratings

Scope (external/internal)

Priority and description


Compliance Engine

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

RAG Pipeline

All structured outputs are embedded into ChromaDB

Vector DB is stored locally (data/vector_db)

No cloud services required

No external accounts needed

Embedded Sources

Mismatches

Project rules

Product data

Documents

Regulations

Query Interface

Engineers can ask questions such as:

“Which walls violate fire safety?”

“Why is wall X non-compliant?”

“What fire rating is required?”

“Which product can resolve this issue?”

“Which regulation supports this rule?”

The system retrieves verified evidence and explains it clearly.

LLM Integration Philosophy

LLMs are used only for explanation

They receive retrieved context from RAG

They are instructed not to invent facts

All data can run locally (e.g., Ollama)

This ensures:

No hallucinations

No compliance risk

Full data privacy


Tech Stack

Python

ifcopenshell

pandas

PyPDF2

ChromaDB

sentence-transformers

Ollama (optional, local LLM)

Design Principles

Deterministic before generative

Explainable by design

Separation of logic and language

Privacy-first (local execution)

Engineering-grade correctness
