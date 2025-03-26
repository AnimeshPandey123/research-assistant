import os
import streamlit as st
from paper_process import fetch_papers, download_and_process_paper
from chat_chain import create_chat_chain

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
    if "dirpath" not in st.session_state:
        st.session_state.dirpath = ""
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
            [ "phi3:14b", "llama3.1:8b", "deepseek-r1", "llama3.2:latest", "mistral"],
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
        
        Papers are only downloaded when you select them for processing, saving bandwidth and storage.
        You can have a continuous conversation about their contents after adding them to your knowledge base.
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
            
            # Fetch papers (metadata only)
            try:
                dirpath, paper_info = fetch_papers(search_query, max_results=max_papers, progress_bar=progress_bar)
                st.session_state.paper_info = paper_info
                st.session_state.dirpath = dirpath
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                progress_bar.empty()
    
    # Paper selection section
    if st.session_state.paper_info:
        with main_container:
            st.subheader(f"📄 Papers on '{st.session_state.topic}'")
            
            # Add Select All button
            col_select_all, col_space = st.columns([1, 3])
            with col_select_all:
                select_all = st.button("Download and Process All Papers", type="primary")
            
            st.write("Select papers to download and add to your knowledge base:")
            
            # Handle Select All functionality
            if select_all:
                for i, paper in enumerate(st.session_state.paper_info):
                    if not paper["processed"]:
                        with st.spinner(f"Downloading and processing '{paper['title']}'..."):
                            retriever, error = download_and_process_paper(
                                st.session_state.paper_info[i],
                                st.session_state.dirpath,
                                st.session_state.topic,
                                st.session_state.retriever
                            )
                            
                            if error:
                                st.error(error)
                            else:
                                # Update retriever and paper status
                                st.session_state.retriever = retriever
                                st.session_state.paper_info[i]["processed"] = True
                                st.session_state.processed_papers.append(i)
                                
                                # Create or update chat chain
                                st.session_state.chat_chain = create_chat_chain(
                                    st.session_state.retriever,
                                    model_name
                                )
                                st.session_state.processing_complete = True
                
                st.success(f"Downloaded and processed {len(st.session_state.paper_info)} papers.")
                st.rerun()
            
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
                        button_text = "Download & Process"
                        if paper.get("downloaded", False):
                            button_text = "Process Paper"
                            
                        if st.button(button_text, key=f"process_btn_{i}"):
                            with st.spinner(f"{'Downloading and processing' if not paper.get('downloaded', False) else 'Processing'} '{paper['title']}'..."):
                                # Download and process this paper
                                retriever, error = download_and_process_paper(
                                    st.session_state.paper_info[i],
                                    st.session_state.dirpath,
                                    st.session_state.topic,
                                    st.session_state.retriever
                                )
                                
                                if error:
                                    st.error(error)
                                else:
                                    # Update retriever and paper status
                                    st.session_state.retriever = retriever
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
            st.info("👆 Select a paper to download and process it to start asking questions.")

if __name__ == "__main__":
    main()