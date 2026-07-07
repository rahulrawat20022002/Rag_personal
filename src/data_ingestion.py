import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingestion_pipeline(folder_path):
    print(f"--- 1. Starting Batch Ingestion for Folder: {folder_path} ---")

    # A. Validation 
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    # B. Load ALL PDFs from the folder (Alle PDFs laden)
    # PyPDFDirectoryLoader scans the folder and loads every PDF it finds.
    print(f"   > Scanning folder '{folder_path}' for PDFs...")
    loader = PyPDFDirectoryLoader(folder_path)
    documents = loader.load()
    
    if not documents:
        print("   > No PDFs found in the directory.")
        return
        
    print(f"   > Loaded {len(documents)} pages from all PDFs combined.")

    # C. Data Parsing & Splitting 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   > Split into {len(chunks)} chunks.")

    # D. Embedding (Einbettung) - Open Source
    print("   > Loading Open Source Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # E. Storing in Vector DB 
    # This adds the new data to the existing DB if it exists, or creates a new one.
    print("   > Updating Vector DB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db_local"
    )
    
    print("--- Ingestion Complete. All folder data stored. ---")

# Run the pipeline
if __name__ == "__main__":
    # Ensure you create a folder named 'my_pdfs' and put your files inside
    ingestion_pipeline("/home/lenovo/Desktop/Rag_pers/data")