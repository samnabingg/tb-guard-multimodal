import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Shared embeddings instance
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ─────────────────────────────────────────────
# STEP 1 — Load all PDFs from the documents folder
# ─────────────────────────────────────────────
def load_documents(folder_path="documents"):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            all_docs.extend(loader.load())
    print(f"\n✅ Total pages loaded: {len(all_docs)}\n")
    return all_docs


# ─────────────────────────────────────────────
# STEP 2 — Split documents into small chunks
# ─────────────────────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = ~500 characters
        chunk_overlap=50      # small overlap so context isn't lost
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}\n")
    return chunks


# ─────────────────────────────────────────────
# STEP 3 — Store chunks in ChromaDB
# ─────────────────────────────────────────────
def build_vector_store(chunks):
    print("⚙️  Building vector store (this may take a minute)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("✅ Vector store built and saved to chroma_db\n")
    return vector_store


# ─────────────────────────────────────────────
# STEP 4 — Search the vector store (used by agents)
# ─────────────────────────────────────────────
def retrieve(query, vector_store, k=4):
    results = vector_store.similarity_search(query, k=k)
    return results


# ─────────────────────────────────────────────
# STEP 5 — Load existing vector store (so you
#           don't rebuild every time)
# ─────────────────────────────────────────────
def load_vector_store():
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vector_store


# ─────────────────────────────────────────────
# RUN — Build the pipeline and test it
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build the pipeline
    docs   = load_documents()
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)

    # Test it with a sample question
    print("🔍 Test query: 'What are the radiological signs of TB?'\n")
    results = retrieve("What are the radiological signs of TB?", vector_store)

    for i, result in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(result.page_content)
        print()



    results2 = retrieve("What is the WHO recommended TB treatment duration?", vector_store)
    for i, r in enumerate(results2):
        print(f"--- Result {i+1} ---")
        print(r.page_content)
        print()