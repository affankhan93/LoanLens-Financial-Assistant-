from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()


DATA_PATH = 'DATA'

# Data load 
def load_pdfs(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    return documents

documents = load_pdfs(DATA_PATH)
print(f"Number of documents are created:- {len(documents)}")

# From documents chunks is created
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    all_splits = text_splitter.split_documents(extracted_data)
    return all_splits

chunks = create_chunks(extracted_data=documents)

# From chunks embeddings is created.
def vector_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embedding = vector_embeddings()
print(f"Vector embeddings are created")

# Chunks, Embeddings, Index are store in vector store.
index_name = "loan-approval-system" 
def vector_database():
    pc = Pinecone(api_key= os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(embedding=embedding, index=index) 
    vector_store.add_documents(documents=chunks)


vector_database() 
print(f"Indexed {len(chunks)} chunks.")


