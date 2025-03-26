import os
import time
import arxiv
import qdrant_client
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

from paper_process import fetch_papers

def process_papers(dirpath, query):
    print(f"Processing papers in {dirpath} for query: {query}...")
    collection_name = f"arxiv_{query.replace(' ', '_')}"
    collection_path = f"./tmp/{collection_name}"
    
    loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
    papers = loader.load()
    
    full_text = " ".join(paper.page_content for paper in papers if paper.page_content)
    print(f"Total characters in {query} text: {len(full_text)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])
    print(f"Split text into {len(paper_chunks)} chunks.")
    
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path=collection_path,
        collection_name=collection_name
    )
    print("Vector database created successfully.")
    
    return qdrant.as_retriever()

def answer_question(retriever, question):
    print(f"Processing question: {question}")
    llm = OllamaLLM(model="phi3:14b")  
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.invoke(question)
    print("Answer retrieved.")
    return response

def main():
    query = input("Enter research topic: ")
    print("Starting paper retrieval and processing...")
    dirpath, _ = fetch_papers(query)
    retriever = process_papers(dirpath, query)
    
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Exiting application.")
            break
        answer = answer_question(retriever, question)
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    main()