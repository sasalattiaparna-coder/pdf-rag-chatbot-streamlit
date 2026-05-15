from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS


# Load PDF
def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


# Split Documents
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    return docs


# Create Vector Store
def create_vectorstore(documents, api_key):

    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore