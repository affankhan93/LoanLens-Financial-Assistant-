from langchain_classic.chains import RetrievalQA 
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()

# Load Groq LLM 
def load_llm():
    llm = ChatGroq(
        api_key= os.environ.get("GROQ_API_KEY"),
        model = 'openai/gpt-oss-120b',
        temperature=0.5,
        max_tokens=1024,
    )
    return llm 


# Created custom prompt that will guide the llm to answer properly 
def create_custom_prompt():
    CUSTOM_PROMPT = """
       You are expert of financial assistant you're job is to answer the quesitons based on the provided content. 
       Don't Hallucinate and remember if you don't know the answer Just say:- "I don't known I'm only trained on Finanacial data."
       Context : {context},
       Question : {question}
       Start with directly no small talk and provide a concise answer based on the context. 
    """

    prompt = PromptTemplate(
        template= CUSTOM_PROMPT,
        input_variables=['context', 'question'],
    )

    return prompt


# Creating Vectore database. 
def create_vectorstore(index_name = 'loan-approval-system'):
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index = pc.Index(index_name)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    vector_store = PineconeVectorStore(embedding = embeddings, index = index)
    return vector_store


# Retrieval the embddings & vector database.
def create_retriever(index_name = 'loan-approval-system', vector_store = None):
    if vector_store is None:
         vector_store = create_vectorstore(index_name)
    retriever = vector_store.as_retriever(search_kwargs = {'k' : 3})
    return retriever 


# Created Question and Answer chain 
def qa_chain(vector_store = None):
    llm = load_llm()
    retriever = create_retriever(vector_store=vector_store)
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = 'stuff',
        return_source_documents = True,
        chain_type_kwargs = {'prompt' : create_custom_prompt()}
    )

    return qa_chain

