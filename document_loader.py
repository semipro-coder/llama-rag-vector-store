# document_loader.py
from dotenv import load_dotenv
import os

load_dotenv()
load_dotenv(dotenv_path=".env")  # Specify the exact path to the .env file
user_agent = os.getenv("USER_AGENT")
os.environ["USER_AGENT"] = user_agent

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(urls):
    """
    Loads documents from the given URLs and splits them into chunks.
    
    Args:
        urls (list): List of URLs to load documents from.

    Returns:
        list: List of document chunks.
    """
    # Load documents from the URLs using WebBaseLoader
    docs = [WebBaseLoader(url).load() for url in urls]
    # Flatten the list of loaded documents
    docs_list = [item for sublist in docs for item in sublist]

    # Initialize a text splitter with a chunk size of 250 and no overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    # Split the documents into smaller chunks
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits
