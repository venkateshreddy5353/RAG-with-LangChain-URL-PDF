from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAI
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# HF_API_KEY = os.getenv("HF_API_KEY")

# llm_openai = OpenAI(api_key=OPEN_AI_API_KEY, model="gpt-3.5-turbo")
llm_gemini = ChatGoogleGenerativeAI( google_api_key= GEMINI_API_KEY, model="gemini-pro")
embeddings_open_ai = OpenAIEmbeddings(api_key=OPEN_AI_API_KEY) # OPEN_AI

# embeddings_cohere = CohereEmbeddings(api_key=COHERE_API_KEY,model="embed-multilingual-v3.0") # embed-english-v3.0
# embeddings_hunggingface = HuggingFaceInferenceAPIEmbeddings(api_key=HF_API_KEY, model="sentence-transformers/all-MiniLM-16-v2")


def ask_gemini(prompt):
    """
    Sends a prompt to the Gemini AI model and returns the response content.

    Args:
        prompt (str): The prompt to send to the Gemini AI model.

    Returns:
        str: The response content from the Gemini AI model.
    """
    AI_Respose = llm_gemini.invoke(prompt)
    return AI_Respose.content



def rag_with_url(target_url, prompt):
    """
    Retrieves relevant documents from a target URL and generates an AI response based on the prompt.

    Args:
        target_url (str): The URL of the target document.
        prompt (str): The prompt for generating the AI response.

    Returns:
        str: The generated AI response.

    Raises:
        Any exceptions that may occur during the execution of the function.

    """
    loader = WebBaseLoader(target_url)
    raw_document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )

    splited_document = text_splitter.split_documents(raw_document)

    vector_store = FAISS.from_documents(splited_document, embeddings_open_ai)

    retriever = vector_store.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    final_prompt = prompt + " " + " ".join([doc.page_content for doc in relevant_documents])

    AI_Respose = llm_gemini.invoke(final_prompt)

    return AI_Respose.content



def rag_with_pdf(file_path, prompt):
    
    """
    Performs RAG (Retrieval-Augmented Generation) using a PDF file.

    Args:
        file_path (str): The path to the PDF file.
        prompt (str): The prompt for the RAG model.

    Returns:
        tuple: A tuple containing the AI response content and a list of relevant documents.

    Raises:
        Any specific exceptions that may occur during the execution of the function.

    """
    loader = PyPDFLoader(file_path)
    
    raw_document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0,
        length_function = len
    )

    splited_document = text_splitter.split_documents(raw_document)

    vector_store = FAISS.from_documents(splited_document, embeddings_open_ai)

    retriever = vector_store.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    final_prompt = prompt + " " + " ".join([doc.page_content for doc in relevant_documents])

    AI_Respose = llm_gemini.invoke(final_prompt)

    return AI_Respose.content, relevant_documents