
#this code is same as main.py but doesnt have a memory feature so it cannot store the info of the user.



import os
import sys
from dotenv import load_dotenv

from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

# 1. SETUP
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    sys.exit("❌ Error: GROQ_API_KEY missing.")

# Check DB
if not os.path.exists("./chroma_db_local"):
    sys.exit("❌ Error: DB not found. Run data_ingestion.py first.")

# Initialize Global Objects
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db_local", embedding_function=embedding_model)
web_search_tool = DuckDuckGoSearchRun()

def process_user_query(user_query):
    # Initialize LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # A. ROUTER
    # We stripped out the {history} variable. Now it only looks at the {question}.
    router_prompt = ChatPromptTemplate.from_template("""
    You are an intelligent router. Decide if the question requires searching the 'PDF Database' or the 'Internet'.
    
    Current Question: {question}
    
    Decide based on these rules:
    1. "SEARCH" -> If question is about OECD Reports, Labour Demand, Net-Zero, or specific document content.
    2. "WEB_SEARCH" -> If question is about real-time events, weather, sports, or news.
    3. "DIRECT" -> If question is General ("Hi", "Thanks") or a simple calculation/fact.
    
    Return ONLY one word: "SEARCH", "WEB_SEARCH", or "DIRECT".
    """)
    
    decision = (router_prompt | llm | StrOutputParser()).invoke({
        "question": user_query
    }).strip().upper()
    
    print(f"   [Brain]: {decision}")

    # B. EXECUTE DECISION
    if decision == "WEB_SEARCH": 
        print("   > Searching Internet...")
        web_content = web_search_tool.invoke(user_query)
        
        final_prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the web result.
        
        Web Result: {web_content}
        Question: {question}
        """)
        
        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": user_query, 
            "web_content": web_content
        })

    elif decision == "SEARCH":
        print("   > Searching Local Database...")
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        
        final_prompt = ChatPromptTemplate.from_template("""
        Answer based on the following context only.
        
        Context: {context}
        Question: {question}
        """)
        
        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": context, 
            "question": user_query
        })

    else:
        # DIRECT ANSWER
        final_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question directly.
        
        Question: {question}
        """)
        
        chain = final_prompt | llm | StrOutputParser()
        return chain.invoke({
            "question": user_query
        })

# 3. INTERACTIVE LOOP
if __name__ == "__main__":
    print("\n✅ System Ready (Memory Disabled). Type 'exit' to quit.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]: 
                print("Goodbye!")
                break
            
            # Process directly
            response = process_user_query(user_input)
            
            print(f"AI: {response}")
            print("-" * 30)
            
        except Exception as e:
            print(f"❌ Error: {e}")