import streamlit as st
import tempfile
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from utils import (
    load_pdf,
    split_documents,
    create_vectorstore
)

# Load Environment Variables
load_dotenv()

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Streamlit Page Config
st.set_page_config(
    page_title="Ask My PDF Bot",
    page_icon="📘",
    layout="wide"
)

# Title
st.title("📘 Ask My PDF Bot")
st.markdown("### RAG Chatbot using Streamlit + OpenRouter")

# Sidebar
st.sidebar.title("Settings")

domain = st.sidebar.selectbox(
    "Select Domain",
    ["Education", "Enterprise", "Legal"]
)

st.sidebar.markdown("---")
st.sidebar.write(f"Selected Domain: {domain}")

# Check API Key
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in .env file")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    try:
        # Save Uploaded PDF Temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())

            pdf_path = tmp_file.name

        st.info("Processing PDF...")

        # Load PDF
        documents = load_pdf(pdf_path)

        # Split Documents
        docs = split_documents(documents)

        # Create Vector Store
        vectorstore = create_vectorstore(
            docs,
            OPENROUTER_API_KEY
        )

        st.success("PDF Processed Successfully!")

        # User Question
        query = st.text_input(
            "Ask a Question from PDF"
        )

        if query:

            with st.spinner("Generating Answer..."):

                # Similarity Search
                matched_docs = vectorstore.similarity_search(
                    query,
                    k=3
                )

                # OpenRouter LLM
                llm = ChatOpenAI(
                    model="openai/gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1"
                )

                # QA Chain
                chain = load_qa_chain(
                    llm,
                    chain_type="stuff"
                )

                # Generate Response
                response = chain.invoke({
                    "input_documents": matched_docs,
                    "question": query
                })

                # Display Answer
                st.subheader("Answer")

                st.write(response["output_text"])

        # Delete Temporary PDF
        os.remove(pdf_path)

    except Exception as e:

        st.error(f"Error: {str(e)}")