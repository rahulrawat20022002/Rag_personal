# this code doesnt use LLM to generate output it just simply search from the databse and gives 
# messy result which hard to read and understad



import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def retrieval_pipeline(user_query):
    print(f"--- 2. Starting Retrieval for Query: '{user_query}' ---")

    # A. Initialize Embedding Model
    # MUST be the same model used in Step 1 (Ingestion)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # B. Load the Vector Database
    # We point to the local folder where Step 1 saved the data
    if not os.path.exists("./chroma_db_local"):
        print("Error: Vector DB not found. Run Step 1 first.")
        return

    vector_db = Chroma(
        persist_directory="./chroma_db_local",
        embedding_function=embedding_model
    )

    # C. Perform Similarity Search (Ähnlichkeitssuche)
    # k=3 means "Give me the top 3 best matching chunks"
    print("   > Searching database...")
    results = vector_db.similarity_search(user_query, k=5)

    # D. Display Results
    if results:
        print(f"\nFound {len(results)} relevant passages:\n")
        for i, doc in enumerate(results):
            # Extract metadata (filename comes from 'source')
            source_file = doc.metadata.get('source', 'Unknown File')
            page_num = doc.metadata.get('page', 'Unknown Page')
            
            print(f"--- Result {i+1} ---")
            print(f"Source: {source_file} (Page {page_num})")
            # print(f"Content: \"{doc.page_content[:2000]}...\"") # Preview first 300 chars
            print(f"Content: \"{doc.page_content}")
            print("-" * 50)
    else:
        print("No relevant information found.")

# Run the pipeline
if __name__ == "__main__":
    # Replace this string with a question relevant to your PDFs
    query = "Summarize how the OECD Employment Outlook 2024 links the net-zero transition to changes in labour demand and job displacement risks"
    retrieval_pipeline(query)