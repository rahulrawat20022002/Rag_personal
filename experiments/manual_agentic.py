
# this code uses LLm to answere pdfs queries as well as general queries that can be handled by LLM

# alone without going in to the vectorDB it can answere geneal queries based on the past training data e.g Thanks, Hi etc.

# the only backdrop here is it doesnt uses web search in order to answere queries which model is not trained on.


import os
import sys
from dotenv import load_dotenv

# We only use the basic libraries that we KNOW are working
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD SECRETS
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("❌ Error: GROQ_API_KEY not found.")
    sys.exit(1)

# --- GLOBAL SETUP ---
print("   > Initializing Manual Agent...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists("./chroma_db_local"):
    print("❌ Error: 'chroma_db_local' not found.")
    sys.exit(1)

vector_db = Chroma(
    persist_directory="./chroma_db_local",
    embedding_function=embedding_model
)

# --- THE AGENT LOGIC (The Router) ---

def run_router_agent(user_query):
    print(f"\n--- Processing: '{user_query}' ---")
    
    # STEP 1: THE DECISION (The Brain)
    # We ask the LLM: "Do you need to search the database for this?"
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    router_template = """
    You are an intelligent router. 
    The user is asking a question. You must decide if you need to search the external 'PDF Database' to answer it.
    
    The Database contains: Information about the OECD Employment Outlook, Net-Zero transition, and Labour Demand.
    
    Rules:
    - If the question is about the Report, Labour, Net-Zero, or specific data -> Return "SEARCH"
    - If the question is General (e.g., "Hi", "Capital of France", "Write a poem") -> Return "DIRECT"
    
    Return ONLY the word "SEARCH" or "DIRECT". Do not explain.
    
    Question: {question}
    """
    
    router_prompt = ChatPromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm | StrOutputParser()
    
    print("   > Agent is deciding strategy...")
    decision = router_chain.invoke(user_query).strip().upper()
    print(f"      [Decision]: {decision}")

    # STEP 2: EXECUTION
    if "SEARCH" in decision:
        # --- PATH A: RAG (Retrieval) ---
        print("   > Strategy: Retrieving documents...")
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(user_query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # Final Answer with Context
        final_template = """
        Answer the question based ONLY on the context below.
        Context: {context}
        Question: {question}
        """
        final_prompt = ChatPromptTemplate.from_template(final_template)
        chain = final_prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "question": user_query})
        
    else:
        # --- PATH B: Direct Answer (No Search) ---
        print("   > Strategy: Answering from general knowledge...")
        final_template = """
        You are a helpful assistant. Answer the question directly.
        Question: {question}
        """
        final_prompt = ChatPromptTemplate.from_template(final_template)
        chain = final_prompt | llm | StrOutputParser()
        response = chain.invoke({"question": user_query})

    print("\n" + "="*60)
    print(f"FINAL ANSWER:\n{response}")
    print("="*60)

if __name__ == "__main__":
    # Test 1: Should trigger SEARCH
    run_router_agent("What are the risks of the net-zero transition?")
    
    # Test 2: Should trigger DIRECT
    run_router_agent("What is the capital of France?")