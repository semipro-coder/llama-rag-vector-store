"""
retriever.py
"""

from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    """
    This class converts text into numerical vectors 
    (embeddings) using a pre-trained AI model
    It uses a free, open-source model that runs 
    locally on your computer
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Load a pre-trained model 
        by default uses a small but effective model called MiniLM
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Takes a list of texts and converts each one 
        into a numerical vector
        Shows a progress bar since this might take a while 
        with many documents
        """
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query):
        """
        Converts a single search query into a numerical vector
        No progress bar needed since it's just one piece of text
        """
        return self.model.encode(query, show_progress_bar=False)

def create_retriever(doc_splits, api_key=None):
    """
    This function sets up a system to find relevant 
    documents based on a search query
    It works in 2 steps:
    1. Convert all documents into numerical vectors using 
    LocalEmbeddings
    2. Store these vectors in a simple database (SKLearnVectorStore) 
    that can quickly find similar texts
    
    doc_splits: List of text chunks from your documents
    Returns: A retriever that can find the 4 most relevant 
    document chunks for any query

    Create the embedding converter
    """
    
    embedding = LocalEmbeddings()

    """
    Set up the vector database and return it as a retriever
    """
    from langchain_community.vectorstores import SKLearnVectorStore
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding,
    )
    return vectorstore.as_retriever(k=4)  
    """
    k=4 means it returns the 4 most similar documents
    """