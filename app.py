from tempfile import NamedTemporaryFile
import os
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader 
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Whitepaper Analysis",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload your white paper", type=['pdf'])


# Proceed only if a file is uploaded
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        document_path = tmp.name
    
    # Inform the user that the whitepaper is being processed
    st.text("Processing the whitepaper...")

    # Initialize the OpenAI model
    llm = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        model="gpt-3.5-turbo",
        temperature=0.0
    )

    # Load the whitepaper content
    with st.spinner("Reading the whitepaper..."):
        reader = PDFReader(document_path)
        whitepaper_text = reader.load_data(document_path)

    # Clean up: remove the temporary file
    os.remove(document_path)

# Ask the user for a question
user_question = st.text_input("What would you like to know about the whitepaper?")

# If there is a question, process it
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.spinner("Generating an answer..."):
        # Use the OpenAI API to get a response
        response = st.session_state.chat_engine.stream_chat(prompt)
        prompt=f"{whitepaper_text}\n\nQuestion: {user_question}\nAnswer:",
        
        answer = response.choices[0].text.strip()  # Extract the answer text
        st.session_state.messages.append({"role": "assistant", "content": answer})

    
# Display previous messages
for message in st.session_state.messages:
    st.text(f"{message['role']}: {message['content']}")
