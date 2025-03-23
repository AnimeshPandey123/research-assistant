import os
import time
import arxiv
import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from search import search_arxiv_papers
import requests
from langchain.prompts import PromptTemplate


def download_pdf(url, dirpath):
    response = requests.get(url)
    # response.raise_for_status()

    if response.status_code != 200:
        print(f"Failed to download {url}. Status code: {response.status_code}")
        return None
    
    os.makedirs(dirpath, exist_ok=True)

    
    # Extract filename from URL
    filename = os.path.basename(url)
    if not filename.endswith(".pdf"):
        filename += ".pdf"  # Ensure it has a .pdf extension
    
    # Create full file path
    file_path = os.path.join(dirpath, filename)
    
    with open(file_path, 'wb') as file:
        file.write(response.content)
    
    return file_path  # Return the saved file path



def fetch_papers(query, max_results=10, progress_bar=None):
    """Fetch papers from arXiv based on the query"""
    results = search_arxiv_papers(query=query,index_name="arxiv_v1", size=max_results)    
    paper_info = []
    dirpath = f"papers/arxiv_papers_{query.replace(' ', '_')}"

    for i, result in enumerate(results['results']):
        if progress_bar:
            progress_bar.progress(min((i + 0.5) / len(results), 1.0))
            
        while True:
            try:
                url = result['url']

                filename =download_pdf(url=url, dirpath=dirpath)
                if filename:
                    # Extract just the filename without path
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
                st.error(f"Error downloading paper: {e}")
                time.sleep(2)
                
        if progress_bar:
            progress_bar.progress(min((i + 0.5) / len(results), 1.0))
    
    return dirpath, paper_info


def process_paper(paper_path, query, existing_retriever=None):
    """Process a single paper and add it to the retriever"""
    collection_name = f"arxiv_{query.replace(' ', '_')}"
    collection_path = f"./tmp/{collection_name}"
    
    # Load the paper
    loader = PyPDFLoader(paper_path)
    pages = loader.load()
    paper_text = " ".join(page.page_content for page in pages if page.page_content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([paper_text])

    # Create a new retriever or add to existing one
    if existing_retriever is None:
        qdrant = Qdrant.from_documents(
            documents=paper_chunks,
            embedding=GPT4AllEmbeddings(),
            path=collection_path,
            collection_name=collection_name
        )
        return qdrant.as_retriever()
    else:
        # Get the underlying vectorstore
        vectorstore = existing_retriever.vectorstore
        # Add new documents
        vectorstore.add_documents(paper_chunks)
        # Return the existing retriever (now updated)
        return existing_retriever


def create_chat_chain(retriever, model_name):
    """Create a conversational chain that maintains chat history"""
    llm = OllamaLLM(model=model_name)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create a custom prompt template that instructs the model to be accurate and stay within context
    prompt_template = """
    Answer the question based ONLY on the following context. If you don't know the answer or the information is not in the context, say "I don't have enough information in the papers to answer this question" instead of making up an answer.

    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    
    Answer:
    """
    
    # Create the prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )
            
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    
    return chain


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
    if "paper_info" not in st.session_state:
        st.session_state.paper_info = []
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "processed_papers" not in st.session_state:
        st.session_state.processed_papers = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "Select Ollama Model:",
            ["mistral", "phi3:14b", "llama3.1:8b", "llama3.2:latest", "deepseek-r1"],
            index=0
        )
        
        max_papers = st.slider("Maximum Papers to Fetch", 1, 20, 2)
        
        # Add a clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.success("Conversation cleared!")
            
        # Add reset knowledge base button
        if st.button("Reset Knowledge Base"):
            st.session_state.processed_papers = []
            st.session_state.retriever = None
            st.session_state.chat_chain = None
            st.session_state.processing_complete = False
            st.session_state.messages = []
            # Reset processed status for all papers
            for paper in st.session_state.paper_info:
                paper["processed"] = False
            st.success("Knowledge base reset! You can now process papers again.")
        
        st.header("About")
        st.write("""
        This app allows you to search for research papers on arXiv and ask questions about them.
        The papers are processed using LangChain and indexed for quick retrieval.
        
        You can select which papers to include in your QA system one at a time and have a continuous conversation about their contents.
        """)
    
    # Create a container for the main interface
    main_container = st.container()
    
    # Search input
    with main_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Enter Research Topic:", key="search_box")
        with col2:
            search_button = st.button("Search Papers", type="primary")
    
    # Process the search when the button is clicked
    if search_button and search_query:
        st.session_state.topic = search_query
        st.session_state.processed_papers = []
        st.session_state.retriever = None
        st.session_state.chat_chain = None
        st.session_state.processing_complete = False
        st.session_state.messages = []  # Clear messages when starting a new search
        
        with st.spinner(f"Fetching papers on '{search_query}'..."):
            progress_bar = st.progress(0)
            
            # Fetch papers
            try:
                dirpath, paper_info = fetch_papers(search_query, max_results=max_papers, progress_bar=progress_bar)
                st.session_state.paper_info = paper_info
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                progress_bar.empty()
    
    # Paper selection section
    if st.session_state.paper_info:
        with main_container:
            st.subheader(f"📄 Papers on '{st.session_state.topic}'")
            st.write("Select and process papers one by one to add to your knowledge base:")
            
            # List papers with process buttons
            for i, paper in enumerate(st.session_state.paper_info):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{paper['title']}** ({paper['year']}) - {paper['authors']}")
                with col2:
                    # Show status or process button
                    if paper["processed"]:
                        st.success("✅ Added to KB")
                    else:
                        if st.button(f"Process Paper", key=f"process_btn_{i}"):
                            with st.spinner(f"Processing '{paper['title']}'..."):
                                # Process this paper
                                st.session_state.retriever = process_paper(
                                    paper["path"],
                                    st.session_state.topic,
                                    st.session_state.retriever
                                )
                                
                                # Update paper status
                                st.session_state.paper_info[i]["processed"] = True
                                st.session_state.processed_papers.append(i)
                                
                                # Create or update chat chain
                                st.session_state.chat_chain = create_chat_chain(
                                    st.session_state.retriever,
                                    model_name
                                )
                                st.session_state.processing_complete = True
                                st.success(f"Added '{paper['title']}' to your knowledge base")
                                # Force a rerun to update UI
                                st.rerun()
    
    # Chat interface section
    if st.session_state.processed_papers:
        with main_container:
            st.divider()
            st.subheader("💬 Chat with the Research Papers")
            
            # Display active papers
            num_papers = len(st.session_state.processed_papers)
            with st.expander(f"Currently using {num_papers} paper{'s' if num_papers > 1 else ''} in your knowledge base:", expanded=False):
                active_papers = [st.session_state.paper_info[i]["title"] for i in st.session_state.processed_papers]
                for title in active_papers:
                    st.write(f"- {title}")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            user_question = st.chat_input("Ask a question about the papers...")
            
            if user_question:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_question})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.chat_chain.invoke({"question": user_question})
                            answer = response.get("answer", "I couldn't find an answer to that question in the papers.")
                            message_placeholder.markdown(answer)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            error_msg = f"An error occurred: {str(e)}"
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Initial guidance
    if not st.session_state.paper_info and not search_button:
        with main_container:
            st.info("👆 Enter a research topic above to get started.")
    elif not st.session_state.processed_papers and st.session_state.paper_info:
        with main_container:
            st.info("👆 Process at least one paper to start asking questions.")


if __name__ == "__main__":
    main()