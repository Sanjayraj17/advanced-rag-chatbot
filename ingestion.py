import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# SETTINGS
# ============================================================

CHROMA_PATH = "./chroma_db_multi"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_BATCH_SIZE = 2000

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model in ingestion...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
print("Embedding model loaded.")

# ============================================================
# CHROMA DATABASE
# ============================================================

print("Connecting to ChromaDB in ingestion...")
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    collection_name="advanced_rag_documents",
    embedding_function=embeddings
)
print("ChromaDB connected.")

# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# ============================================================
# ADD PDF TO CHROMA
# ============================================================

def add_pdf(pdf_path):
    file_name = os.path.basename(pdf_path)

    print("\n" + "=" * 60)
    print(f"Adding PDF: {file_name}")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return 0

    # Remove existing chunks for this PDF to avoid duplicates
    print(f"Checking for existing chunks of {file_name}...")
    delete_pdf(file_name)

    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------
    print("\nLoading PDF...")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Failed to load PDF: {e}")
        return 0

    if not documents:
        print("Warning: PDF contains no readable pages.")
        return 0

    print("PDF loaded successfully.")
    print(f"Pages: {len(documents)}")

    # --------------------------------------------------------
    # SPLIT PDF
    # --------------------------------------------------------
    print("\nSplitting PDF into chunks...")
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        print("Warning: No text chunks extracted from PDF.")
        return 0

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------------------------
    # ADD SOURCE NAME (NORMALIZED BASENAME)
    # --------------------------------------------------------
    for chunk in chunks:
        chunk.metadata["source"] = file_name

    # --------------------------------------------------------
    # ADD TO CHROMA IN SAFE BATCHES (<= 2000)
    # --------------------------------------------------------
    print("\nCreating embeddings and adding to ChromaDB...")
    for i in range(0, len(chunks), MAX_BATCH_SIZE):
        batch = chunks[i:i + MAX_BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Indexed chunks {i + 1} to {min(i + MAX_BATCH_SIZE, len(chunks))}")

    print("\nPDF added to ChromaDB successfully.")
    print(f"File: {file_name}")
    print(f"Chunks added: {len(chunks)}")
    print("=" * 60)

    return len(chunks)

# ============================================================
# DELETE PDF FROM CHROMA
# ============================================================

def delete_pdf(file_name):
    base_name = os.path.basename(file_name)

    print("\n" + "=" * 60)
    print(f"Deleting PDF: {base_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # FIND ALL CHUNKS BELONGING TO THIS PDF
    # --------------------------------------------------------
    print("\nFinding PDF chunks...")
    all_data = vectorstore.get()
    
    ids_to_delete = []
    if all_data and "ids" in all_data and "metadatas" in all_data:
        for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            if meta and "source" in meta:
                source_val = meta["source"]
                if os.path.basename(source_val) == base_name or source_val == base_name:
                    ids_to_delete.append(chunk_id)

    if not ids_to_delete:
        print("\nNo chunks found for this PDF.")
        return 0

    print(f"Found {len(ids_to_delete)} chunks.")
    print("\nDeleting chunks from ChromaDB...")

    # Delete in safe batch sizes (<= 2000)
    for i in range(0, len(ids_to_delete), MAX_BATCH_SIZE):
        batch_ids = ids_to_delete[i:i + MAX_BATCH_SIZE]
        vectorstore.delete(ids=batch_ids)

    print("\nPDF removed from ChromaDB successfully.")
    print(f"Chunks removed: {len(ids_to_delete)}")
    print("=" * 60)

    return len(ids_to_delete)