import os
from ingestion import add_pdf

data_dir = "data"
available_pdfs = [
    os.path.join(data_dir, f) for f in os.listdir(data_dir)
    if f.lower().endswith(".pdf")
]

if not available_pdfs:
    print("No PDF files found in data/ directory to test.")
else:
    # Pick the smallest file to test quickly
    test_pdf = min(available_pdfs, key=lambda f: os.path.getsize(f))
    print(f"Testing ingestion with: {test_pdf}")
    chunks_added = add_pdf(test_pdf)
    print(f"\nSuccessfully added/re-indexed {chunks_added} chunks.")