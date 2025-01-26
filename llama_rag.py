# llama_rag.py

from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from document_loader import load_and_split_documents
from retriever import create_retriever
from dotenv import load_dotenv
import os

load_dotenv() 


# Define the prompt template for the language model
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",  
    temperature=0,     
)

# Combine the prompt and the LLM into a single chain
rag_chain = prompt | llm | StrOutputParser()

class RAGApplication:
    """
    RAG (Retrieval-Augmented Generation) application for question-answering tasks.
    """
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        """
        Answers a question using retrieved documents and the language model.

        Args:
            question (str): The question to answer.

        Returns:
            str: The generated answer.
        """
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from the retrieved documents
        doc_texts = "\n".join([doc.page_content for doc in documents])
        # Get the answer from the LLM
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

# Main script execution
if __name__ == "__main__":
    # URLs to load documents from
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load and split documents
    doc_splits = load_and_split_documents(urls)

    # Create the retriever
    api_key = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key
    retriever = create_retriever(doc_splits, api_key)

    # Initialize the RAG application
    rag_application = RAGApplication(retriever, rag_chain)

    # Example question
    question = "What is prompt engineering?"
    # Run the RAG application
    answer = rag_application.run(question)

    question2 = "What are types of attacks on LLMs?"
    answer2 = rag_application.run(question2)

    question3 = "What is the square root of pi?"
    answer3 = rag_application.run(question3)

    # Print the result
    print("Question:", question)
    print("Answer:", answer)
    print("Question2:", question2)
    print("Answer2:", answer2)
    print("Question3:", question3)
    print("Answer3:", answer3)
