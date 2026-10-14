import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. FIND PDF FILES
# =========================================================

data_folder = "data"

pdf_files = []

for file in os.listdir(data_folder):

    if file.lower().endswith(".pdf"):

        pdf_files.append(
            os.path.join(data_folder, file)
        )


print("PDF files found:")

for pdf in pdf_files:

    print("-", pdf)


# =========================================================
# 2. LOAD ALL PDFs
# =========================================================

all_documents = []

for pdf in pdf_files:

    print("\nLoading:", pdf)

    loader = PyMuPDFLoader(pdf)

    documents = loader.load()

    all_documents.extend(documents)

    print("Pages loaded:", len(documents))


# =========================================================
# 3. SPLIT ALL PDFs
# =========================================================

print("\nSplitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    all_documents
)

for chunk in chunks:
    if "source" in chunk.metadata:
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

print("Total chunks:", len(chunks))


# =========================================================
# 5. SHOW PDF SOURCES
# =========================================================

print("\nSources:")

for i, chunk in enumerate(chunks[:10]):

    print(
        f"Chunk {i + 1}:",
        chunk.metadata.get("source")
    )