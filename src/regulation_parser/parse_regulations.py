from pathlib import Path
import json
from PyPDF2 import PdfReader


REGULATIONS_DIR = "data/regulations"
OUTPUT_PATH = "data/output/regulations.json"

# 🔧 CONFIG — adjust safely
START_PAGE = 300      # zero-based index (page 301)
END_PAGE = 600        # page 601
KEYWORDS = ["fire", "wall", "partition", "ceiling", "EI", "REI"]


def extract_relevant_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""

    for i in range(START_PAGE, min(END_PAGE, len(reader.pages))):
        page_text = reader.pages[i].extract_text()
        if not page_text:
            continue

        page_text_lower = page_text.lower()
        if any(keyword.lower() in page_text_lower for keyword in KEYWORDS):
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
    regulations = []

    for pdf in Path(REGULATIONS_DIR).glob("*.pdf"):
        text = extract_relevant_text(str(pdf))
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            regulations.append({
                "reg_id": f"{pdf.stem.upper()}_{i+1}",
                "source": pdf.name,
                "content": chunk
            })

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(regulations, f, indent=2)

    print(f"LAYER 4 COMPLETE — {len(regulations)} filtered regulation chunks created")


if __name__ == "__main__":
    main()
