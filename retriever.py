from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    """
    Embedding class that wraps sentence-transformers to provide embeddings
    for both documents and queries.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load the sentence-transformers model
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Embed a list of document texts.

        Args:
            texts (list): List of document texts.

        Returns:
            list: List of embeddings for the documents.
        """
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query):
        """
        Embed a single query text.

        Args:
            query (str): Query text.

        Returns:
            list: Embedding for the query.
        """
        return self.model.encode(query, show_progress_bar=False)

def create_retriever(doc_splits, api_key=None):
    """
    Creates a retriever using SentenceTransformers embeddings.

    Args:
        doc_splits (list): List of document chunks.

    Returns:
        SKLearnVectorStore retriever object.
    """
    # Instantiate LocalEmbeddings
    embedding = LocalEmbeddings()

    # Create the SKLearnVectorStore using the LocalEmbeddings
    from langchain_community.vectorstores import SKLearnVectorStore
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding,
    )
    return vectorstore.as_retriever(k=4)
