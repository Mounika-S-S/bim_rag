import sys
import os

# add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.l4.pdf_cleaner import PDFCleaner
from src.l4.clause_segmenter import ClauseSegmenter
import json

def test():
    cleaner = PDFCleaner(crop_margin=0.08, drop_empty_pages=True, skip_toc=True)
    segmenter = ClauseSegmenter()
    
    pdf_path = r"c:\DRIVE1\ip\bim_rag\data\processed\mouni\upload_L4_l4.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
        
    print("Extracting...")
    cleaned_text = cleaner.clean(pdf_path)
    print("Extracted Length:", len(cleaned_text))
    
    with open("cleaned_output.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text[:5000])

    print("Segmenting...")
    # clause segmenter initializes its own cleaner, but we updated the cleaner code.
    clauses = segmenter.segment(pdf_path)
    
    with open("clauses_output.json", "w", encoding="utf-8") as f:
        json.dump(clauses[:50], f, indent=4)
        
    print("Done")

if __name__ == "__main__":
    test()
