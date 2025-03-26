import os
import time
import requests
from search import search_arxiv_papers

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
    results = search_arxiv_papers(query=query, index_name="arxiv_v1", size=max_results)    
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
