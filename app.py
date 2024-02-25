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
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about your whitepaper!"}
    ]

uploaded_file = st.file_uploader("Upload your white paper", type=['pdf'])

# Proceed only if a file is uploaded
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:#open a named temporary file
        tmp.write(uploaded_file.getvalue()) #write data from uploaded file into it
        document_path = tmp.name
        with st.spinner("Reading the whitepaper..."):
            reader = PDFReader(document_path)
            whitepaper_text = reader.load_data(document_path)
            # Initialize the OpenAI model
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                model="gpt-3.5-turbo",
                temperature=0.0,
                system_prompt="You are an expert on the content of the document, provide detailed answers to the questions. Use the document to support your answers.",
            )
            index = VectorStoreIndex.from_documents(whitepaper_text)
    # Clean up: remove the temporary file
    os.remove(document_path)
    if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=False, llm=llm
        )

#prompt
if prompt := st.chat_input(
    "What would you like to know about the whitepaper?"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history