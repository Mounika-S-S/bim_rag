from pathlib import Path
import json
from PyPDF2 import PdfReader


PDF_PATH = "data/regulations/fire_code_1.pdf"
OUTPUT_PATH = "data/output/regulations.json"


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text: str, max_chars=900):
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

    regulations = []
    for i, chunk in enumerate(chunks):
        regulations.append({
            "reg_id": f"FIRE_CODE_1_{i+1}",
            "source": "fire_code_1.pdf",
            "content": chunk
        })

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(regulations, f, indent=2)

    print(f"LAYER 4 COMPLETE — {len(regulations)} regulation chunks created from fire_code_1.pdf")


if __name__ == "__main__":
    main()
