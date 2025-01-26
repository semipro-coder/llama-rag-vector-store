# This line imports the load_dotenv function which helps us read secret information from a file
from dotenv import load_dotenv
# This imports the os module which lets us work with environment variables and files
import os

# Try to load secret information from a file called .env
load_dotenv()


# Get the USER_AGENT value from our secret file - this is like an ID card for our program
# when it visits websites
user_agent = os.getenv("USER_AGENT")
# Store this USER_AGENT where our program can use it
os.environ["USER_AGENT"] = user_agent

# Import tools we need to download and process web pages
# WebBaseLoader helps us download web pages
from langchain_community.document_loaders import WebBaseLoader
# RecursiveCharacterTextSplitter helps us break big texts into smaller pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(urls):
    """
    This function does two main things:
    1. Downloads web pages from a list of URLs
    2. Breaks these web pages into smaller chunks that are easier to work with
    
    Args:
        urls (list): A list of website addresses we want to download
    
    Returns:
        list: The web pages broken up into smaller pieces
    """
    # For each website address (URL) in our list:
    # 1. Create a WebBaseLoader for that URL
    # 2. Use .load() to download the content
    # This creates a list of downloaded documents
    docs = [WebBaseLoader(url).load() for url in urls]
    
    # The above step gives us a complex list (a list of lists)
    # This line flattens it into a simple list we can work with
    # For example: [[1,2],[3,4]] becomes [1,2,3,4]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Create a tool that will help us split documents into smaller pieces
    # - chunk_size=250 means each piece will be about 250 characters long
    # - chunk_overlap=0 means the pieces won't overlap with each other
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0
    )
    
    # Use our text splitter to break the documents into smaller pieces
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Return our list of small document pieces
    return doc_splits