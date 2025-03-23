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

from search import search_arxiv_papers
import requests

def download_pdf(url, dirpath):
    print(f"Downloading PDF from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    filename = os.path.basename(url)
    if not filename.endswith(".pdf"):
        filename += ".pdf"  # Ensure it has a .pdf extension
    
    os.makedirs(dirpath, exist_ok=True)
    file_path = os.path.join(dirpath, filename)
    
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    print(f"Saved PDF to {file_path}")
    return file_path  # Return the saved file path

def fetch_papers(query, max_results=10, progress_bar=None):
    print(f"Fetching papers for query: {query}...")
    results = search_arxiv_papers(query=query, index_name="arxiv", size=max_results)    
    paper_info = []
    dirpath = f"papers/arxiv_papers_{query.replace(' ', '_')}"
    os.makedirs(dirpath, exist_ok=True)
    
    for i, result in enumerate(results['results']):
        if progress_bar:
            progress_bar.progress((i + 0.5) / len(results))
        
        while True:
            try:
                url = result['url']
                filename = download_pdf(url=url, dirpath=dirpath)
                base_filename = os.path.basename(filename)
                paper_info.append({
                    "title": result['title'],
                    "year": result['year'],
                    "authors": result['authors'],
                    "filename": base_filename,
                    "path": filename,
                    "processed": False
                })
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred, retrying in 2 seconds...")
                time.sleep(2)
        
        if progress_bar:
            progress_bar.progress((i + 1) / len(results))
    
    print(f"Fetched {len(paper_info)} papers and saved to {dirpath}")
    return dirpath, paper_info

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