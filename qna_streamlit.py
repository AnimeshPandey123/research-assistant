import os
import time
import arxiv
import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA


def fetch_papers(query, max_results=2, progress_bar=None):
    """Fetch papers from arXiv based on the query"""
    dirpath = f"papers/arxiv_papers_{query.replace(' ', '_')}"
    os.makedirs(dirpath, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = list(client.results(search))
    paper_titles = []
    
    for i, result in enumerate(results):
        if progress_bar:
            progress_bar.progress((i + 0.5) / len(results))
            
        while True:
            try:
                result.download_pdf(dirpath=dirpath)
                paper_titles.append(f"{result.title} ({result.published.year})")
                break
            except (FileNotFoundError, ConnectionResetError) as e:
                st.error(f"Error downloading paper: {e}")
                time.sleep(2)
                
        if progress_bar:
            progress_bar.progress((i + 1) / len(results))
    
    return dirpath, paper_titles


def process_papers(dirpath, query):
    """Process downloaded papers and create a retriever"""
    collection_name = f"arxiv_{query.replace(' ', '_')}"
    collection_path = f"./tmp/{collection_name}"

    loader = DirectoryLoader(dirpath, glob="*.pdf", loader_cls=PyPDFLoader)
    papers = loader.load()

    full_text = " ".join(paper.page_content for paper in papers if paper.page_content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])

    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path=collection_path,
        collection_name=collection_name
    )

    return qdrant.as_retriever()


def answer_question(retriever, question, model_name):
    """Answer a question using the retriever and LLM"""
    llm = OllamaLLM(model=model_name)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.invoke(question)
    return response["result"] if isinstance(response, dict) and "result" in response else response


# Set up the Streamlit app
def main():
    st.set_page_config(
        page_title="Research Paper QA System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Research Paper QA System")
    st.write("Search for academic papers and ask questions about their content.")
    
    # Initialize session state variables
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "paper_titles" not in st.session_state:
        st.session_state.paper_titles = []
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "Select Ollama Model:",
            ["phi3:14b", "llama3:8b", "mistral:7b", "gemma:7b"],
            index=0
        )
        
        max_papers = st.slider("Maximum Papers to Fetch", 1, 10, 2)
        
        st.header("About")
        st.write("""
        This app allows you to search for research papers on arXiv and ask questions about them.
        The papers are processed using LangChain and indexed for quick retrieval.
        """)
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter Research Topic:", key="search_box")
    with col2:
        search_button = st.button("Search Papers", type="primary")
    
    # Process the search when the button is clicked
    if search_button and search_query:
        st.session_state.topic = search_query
        with st.spinner(f"Fetching papers on '{search_query}'..."):
            progress_bar = st.progress(0)
            
            # Fetch papers
            try:
                dirpath, paper_titles = fetch_papers(search_query, max_results=max_papers, progress_bar=progress_bar)
                st.session_state.paper_titles = paper_titles
                
                # Process papers and create retriever
                with st.spinner("Processing papers and creating database..."):
                    st.session_state.retriever = process_papers(dirpath, search_query)
                    
                st.success(f"Successfully processed {len(paper_titles)} papers on '{search_query}'")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                progress_bar.empty()
    
    # Display fetched papers
    if st.session_state.paper_titles:
        st.subheader(f"📄 Papers on '{st.session_state.topic}'")
        for i, title in enumerate(st.session_state.paper_titles):
            st.write(f"{i+1}. {title}")
    
    # Question answering section
    if st.session_state.retriever:
        st.divider()
        st.subheader("🔍 Ask Questions About the Papers")
        
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        answer = answer_question(st.session_state.retriever, question, model_name)
                        
                        st.subheader("Answer:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"An error occurred while generating the answer: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    # Initial guidance
    if not st.session_state.retriever and not search_button:
        st.info("👆 Enter a research topic above to get started.")


if __name__ == "__main__":
    main()