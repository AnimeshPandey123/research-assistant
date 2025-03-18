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


def fetch_papers(query, max_results=2):
    dirpath = f"papers/arxiv_papers_{query.replace(' ', '_')}"
    os.makedirs(dirpath, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_order=arxiv.SortOrder.Descending
    )

    for result in client.results(search):
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                print(f"-> Paper '{result.title}' downloaded.")
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                print("Error occurred:", e)
                time.sleep(5)
    
    return dirpath


def process_papers(dirpath, query):
    collection_name = f"arxiv_{query.replace(' ', '_')}"
    collection_path = f"./tmp/{collection_name}"

    loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
    papers = loader.load()

    full_text = " ".join(paper.page_content for paper in papers if paper.page_content)
    print(f"Total characters in {query} text:", len(full_text))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])

    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path=collection_path,
        collection_name=collection_name
    )

    return qdrant.as_retriever()


def answer_question(retriever, question):
    llm = OllamaLLM(model="phi3:14b")  # Use local Ollama model (e.g., Mistral)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.invoke(question)
    return response


def main():
    query = input("Enter research topic: ")
    dirpath = fetch_papers(query)
    retriever = process_papers(dirpath, query)
    
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = answer_question(retriever, question)
        print("\nAnswer:", answer, "\n")


if __name__ == "__main__":
    main()
