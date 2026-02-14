import sys
import os
import streamlit as st
from dotenv import load_dotenv

# --- 1. PATH MAGIC (To find 'src' folder) ---
# This tells Python to look 2 folders up (to the root RAG_PERS)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
# --------------------------------------------

# Now we can import from src!
from src.memory_agent import MemoryAgent
from src.main import process_user_query

# 2. SETUP PAGE
st.set_page_config(page_title="Agentic RAG", page_icon="🤖")
st.title("🤖 Intelligent Agent (PDF + Web)")

load_dotenv()

# Verify API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ Error: GROQ_API_KEY missing in .env file.")
    st.stop()

# 3. INITIALIZE MEMORY (Session State)
# We store the memory object inside Streamlit's session_state so it persists
if "memory" not in st.session_state:
    st.session_state.memory = MemoryAgent()

# 4. DISPLAY CHAT HISTORY
for msg in st.session_state.memory.history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 5. INPUT & EXECUTION
if input_text := st.chat_input("Ask about the PDF, Weather, or anything..."):
    
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(input_text)
    
    # B. Add to Memory (User)
    st.session_state.memory.add_user_message(input_text)
    
    # C. Prepare History String
    history_str = st.session_state.memory.get_history_string()
    
    # D. Run the Logic (Imported from main.py)
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                # We reuse the logic from your main.py!
                response = process_user_query(input_text, history_str)
                st.markdown(response)
                
                # E. Add to Memory (AI)
                st.session_state.memory.add_ai_message(response)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")