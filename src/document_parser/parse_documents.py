from pathlib import Path
import json
from PyPDF2 import PdfReader


PDF_PATH = "data/documents/Layer2_Product_Data_EN.pdf"
OUTPUT_PATH = "data/output/documents.json"


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text


def chunk_text(text: str, max_chars=800):
    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) > max_chars:
            chunks.append(current.strip())
            current = ""
        current += line + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


def main():
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "doc_id": f"DOC_{i+1}",
            "source": "Layer2_Product_Data_EN.pdf",
            "content": chunk
        })

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(documents, f, indent=2)

    print(f"LAYER 3 COMPLETE — {len(documents)} document chunks created")


if __name__ == "__main__":
    main()
