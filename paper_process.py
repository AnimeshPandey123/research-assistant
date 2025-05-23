import os
import time
import requests
from search import search_arxiv_papers
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def download_pdf(url, dirpath):
    print(f"Checking PDF at {url}...")

    filename = os.path.basename(url)
    if not filename.endswith(".pdf"):
        filename += ".pdf"  # Ensure it has a .pdf extension
    
    os.makedirs(dirpath, exist_ok=True)
    file_path = os.path.join(dirpath, filename)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}. Skipping download.")
        return file_path  # Return existing file path

    print(f"Downloading PDF from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
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
            progress_bar.progress(min(round((i + 1) / len(results['results']), 2), 1.0))
        
        while True:
            try:
                url = result['url']
                paper_info.append({
                    "title": result['title'],
                    "year": result['year'],
                    "authors": result['authors'],
                    "filename": '',
                    "path": '',
                    "processed": False,
                    "downloaded": False,
                    "url": url
                })
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred, retrying in 2 seconds...")
                time.sleep(2)
        
        if progress_bar:
            progress_bar.progress(min(round((i + 1) / len(results['results']), 2), 1.0))
    
    print(f"Fetched {len(paper_info)} papers and saved to {dirpath}")
    return dirpath, paper_info

def download_and_process_paper(paper_info, dirpath, query, existing_retriever=None):
    """Download a paper and add it to the retriever with enriched metadata for each document chunk"""
    
    # Download the paper if not already downloaded
    if not paper_info["downloaded"]:
        try:
            file_path = download_pdf(paper_info["url"], dirpath)
            if not file_path:
                return None, "Failed to download the paper."
            
            paper_info["path"] = file_path
            paper_info["downloaded"] = True
            # Extract just the filename without path
            paper_info["filename"] = os.path.basename(file_path)
        except Exception as e:
            return None, f"Error downloading paper: {str(e)}"
    
    # Process the paper and add to retriever
    try:
        collection_name = f"arxiv_{query.replace(' ', '_')}"
        collection_path = f"./tmp/{collection_name}"
        
        # Load the paper from PDF
        loader = PyPDFLoader(paper_info["path"])
        pages = loader.load()
        paper_text = " ".join(page.page_content for page in pages if page.page_content)

        print(paper_text)
        
        # Split the paper into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        paper_chunks = text_splitter.create_documents([paper_text])

        # Enrich each chunk with metadata that includes paper details and a unique chunk index
        enriched_chunks = []
        for idx, chunk in enumerate(paper_chunks):
            metadata = {
                "title": paper_info["title"],
                "year": paper_info["year"],
                "authors": paper_info["authors"],
                "chunk_index": idx
            }
            chunk.metadata = metadata
            enriched_chunks.append(chunk)
        
        # Create a new retriever or update an existing one with the enriched chunks
        if existing_retriever is None:
            # Create the Qdrant vector store using the pre-initialized embedding and enriched chunks
            qdrant_store = Qdrant.from_documents(
                documents=enriched_chunks,
                embedding=embedding,
                path=collection_path,
                collection_name=collection_name,
                force_recreate=True
            )
            return qdrant_store.as_retriever(), None
        else:
            # Append new documents to the existing vector store
            vectorstore = existing_retriever.vectorstore
            vectorstore.add_documents(enriched_chunks)
            return existing_retriever, None
    except Exception as e:
        return None, f"Error processing paper: {str(e)}"
