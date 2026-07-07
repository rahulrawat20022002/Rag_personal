
#this code uses LLM to answere query based on the pdfs data and knwoledge which gives output in 
# more understandble and readable form


import os
import sys
from dotenv import load_dotenv

# 1. NEW IMPORTS
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 2. LOAD SECRETS
load_dotenv()

# Verify Keys
if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

def generation_pipeline_stable(user_query):
    print(f"\n--- 3. Generation for: '{user_query}' ---")

    # A. SETUP RETRIEVER (Locally - Unchanged)
    print("   > Loading Local Vector DB...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists("./chroma_db_local"):
        print("Error: 'chroma_db_local' not found. Run Step 1 (Ingestion) first.")
        return

    vector_db = Chroma(
        persist_directory="./chroma_db_local",
        embedding_function=embedding_model
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # B. SETUP MODEL (Updated for 2025)
    # We use 'llama-3.1-8b-instant' which is the official replacement
    print("   > Connecting to Groq Cloud (Llama 3.1)...")
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0.0
        )
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # C. CREATE PROMPT
    template = """
    You are an expert assistant. Answer the question based ONLY on the context below.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # D. BUILD & RUN CHAIN
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("   > Generating Answer...")
    try:
        response = chain.invoke(user_query)
        print("\n" + "="*60)
        print(f"FINAL ANSWER:\n{response}")
        print("="*60)
    except Exception as e:
        print(f"Generation Error: {e}")

if __name__ == "__main__":
    query = "Summarize the key points about labour demand."
    generation_pipeline_stable(query)