import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================================================
# 1. FIND ALL PDFs
# =========================================================

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

pdf_files = [
    os.path.join(data_folder, file)
    for file in os.listdir(data_folder)
    if file.lower().endswith(".pdf")
]

print("PDF files found:")
for pdf in pdf_files:
    print("-", pdf)

if not pdf_files:
    print("No PDF files found in data/ folder.")
    exit(0)

# =========================================================
# 2. LOAD ALL PDFs
# =========================================================

all_documents = []

for pdf in pdf_files:
    print("\nLoading:", pdf)
    try:
        loader = PyMuPDFLoader(pdf)
        documents = loader.load()
        # Normalize source in metadata to base filename
        base_name = os.path.basename(pdf)
        for doc in documents:
            doc.metadata["source"] = base_name
        all_documents.extend(documents)
        print("Pages loaded:", len(documents))
    except Exception as e:
        print(f"Error loading {pdf}: {e}")

# =========================================================
# 3. SPLIT DOCUMENTS
# =========================================================

print("\nSplitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(all_documents)
print("Total chunks:", len(chunks))

# Ensure source metadata is strictly basename
for chunk in chunks:
    if "source" in chunk.metadata:
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])

# =========================================================
# 4. LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# =========================================================
# 5. CREATE / UPDATE MULTI-PDF CHROMADB IN BATCHES
# =========================================================

print("\nCreating ChromaDB...")
persist_dir = "./chroma_db_multi"
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)

BATCH_SIZE = 2000
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    vectorstore.add_documents(batch)
    print(f"Added batch {i + 1} to {min(i + BATCH_SIZE, len(chunks))}")

print("ChromaDB created successfully!")

# =========================================================
# 6. DISPLAY INFORMATION
# =========================================================

print("\nDatabase information:")
print("PDFs:", len(pdf_files))
print("Pages:", len(all_documents))
print("Chunks:", len(chunks))
print("\nMulti-PDF ChromaDB is ready.")