import os
import sys

# --- PATH FIXER FOR DATABASE ---
# This ensures we find the DB folder whether running from root, src, or app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # Points to RAG_PERS/
# -------------------------------

from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

# Import Memory (Try/Except for robustness)
try:
    from src.memory_agent import MemoryAgent
except ImportError:
    try:
        from memory_agent import MemoryAgent
    except ImportError:
        class MemoryAgent: pass

# 1. SETUP GLOBAL OBJECTS
load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
web_search_tool = DuckDuckGoSearchRun()

# --- ROBUST DB LOADING ---
# We don't use sys.exit() here because it kills Streamlit.
# Instead, we try to find the path and load it safely.
possible_paths = [
    os.path.join(parent_dir, "chroma_db_local"),       # RAG_PERS/chroma_db_local
    "./chroma_db_local",                               # Root execution
    "../chroma_db_local"                               # src execution
]

db_path = None
for path in possible_paths:
    if os.path.exists(path):
        db_path = path
        break

if db_path:
    vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
else:
    vector_db = None  # We handle this inside the function now


# 2. THE LOGIC FUNCTION (Updated to accept history!)
def process_user_query(user_query, history_string):
    """
    Main Logic: Accepts Question + History -> Returns Answer.
    """
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # A. ROUTER
    router_prompt = ChatPromptTemplate.from_template("""
    You are an intelligent router. Decide if the question requires searching the 'PDF Database' or the 'Internet'.
    
    Previous Conversation:
    {history}
    
    Current Question: {question}
    
    Rules:
    1. "SEARCH" -> If question is about OECD Reports, Labour Demand, Net-Zero.
    2. "WEB_SEARCH" -> If question is about real-time events, weather, sports, or news.
    3. "DIRECT" -> If question is General ("Hi", "Thanks") or refers to history.
    
    Return ONLY one word: "SEARCH", "WEB_SEARCH", or "DIRECT".
    """)
    
    try:
        decision = (router_prompt | llm | StrOutputParser()).invoke({
            "question": user_query, 
            "history": history_string
        }).strip().upper()
    except Exception as e:
        return f"Error in Router: {e}"
    
    print(f"   [Brain]: {decision}")

    # B. EXECUTE
    if decision == "WEB_SEARCH": 
        print("   > Searching Internet...")
        try:
            web_content = web_search_tool.invoke(user_query)
        except Exception as e:
            web_content = f"Search failed: {e}"
        
        final_prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the web result.
        Previous Chat: {history}
        Web Result: {web_content}
        Question: {question}
        """)
        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": user_query, 
            "web_content": web_content, 
            "history": history_string
        })

    elif decision == "SEARCH":
        print("   > Searching Local Database...")
        if not vector_db:
            return "Error: Database not found. Please run data ingestion first."
            
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        
        final_prompt = ChatPromptTemplate.from_template("""
        Answer based on context.
        Previous Chat: {history}
        Context: {context}
        Question: {question}
        """)
        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": context, 
            "question": user_query, 
            "history": history_string
        })

    else:
        # DIRECT
        final_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer based on your knowledge and the conversation history.
        Previous Chat: {history}
        Question: {question}
        """)
        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": user_query, 
            "history": history_string
        })

# 3. INTERACTIVE LOOP (CLI ONLY)
if __name__ == "__main__":
    # Checks specific to running in Terminal
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("❌ Error: GROQ_API_KEY missing.")
    
    # We initialize Memory ONLY when running as main script
    local_memory = MemoryAgent()
    
    print("\n✅ System Ready (CLI Mode)! (Type 'exit' to quit)")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            # 1. Get History String
            hist_str = local_memory.get_history_string()
            
            # 2. Add User to Memory
            local_memory.add_user_message(user_input)
            
            # 3. Process (Pass history string!)
            response = process_user_query(user_input, hist_str)
            
            # 4. Add AI to Memory
            local_memory.add_ai_message(response)
            
            print(f"AI: {response}")
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ Error: {e}")