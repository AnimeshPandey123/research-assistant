# 📚 Research Paper QA System

A powerful tool that allows you to search for academic papers on arXiv, download them, and ask questions about their content using advanced language models and semantic search.

## Overview

This application enables researchers and students to:
- Search for academic papers on specific topics using arXiv
- Select papers of interest to download and process
- Build a personal knowledge base from multiple papers
- Ask questions and receive answers based on the content of the processed papers
- Engage in continuous conversation about the research content

## Features

- **Paper Search**: Find relevant papers on arXiv using Elasticsearch with advanced query expansion
- **Selective Processing**: Choose which papers to add to your knowledge base
- **PDF Processing**: Automatically download and extract content from PDFs
- **Vector Embedding**: Convert paper content into searchable vector embeddings
- **Conversational QA**: Ask questions in natural language about paper content
- **Multiple Model Support**: Choose from various Ollama models (phi4, llama3, mistral, etc.)
- **Interactive UI**: Clean Streamlit interface with search, paper selection, and chat capabilities

## Technical Components

- **Frontend**: Built with Streamlit for an interactive web interface
- **Search**: Elasticsearch for robust paper discovery
- **Document Processing**: LangChain for PDF loading and text splitting
- **Vector Storage**: Qdrant for efficient retrieval of relevant content
- **Embeddings**: HuggingFace sentence transformers for text embeddings
- **LLM Integration**: Support for various Ollama models for answering questions

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Ollama installed with supported models

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AnimeshPandey123/research-assistant.git
   cd research-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Elasticsearch using Docker Compose:
   ```bash
   docker-compose up -d elasticsearch
   ```

5. Download the arXiv dataset:
   ```bash
   # Create dataset directory
   mkdir -p dataset
   
   # Download from Kaggle and put unzipped file in dataset directory
   # https://www.kaggle.com/datasets/Cornell-University/arxiv
   ```

6. Index the arXiv dataset into Elasticsearch:
   ```bash
   python index.py
   ```

7. Ensure Ollama is installed and running with the required models:
   ```bash
   # Example: Pull a model with Ollama
   ollama pull phi4
   ```

## Docker Compose Configuration

The system uses Docker Compose to run Elasticsearch. Below is the configuration:

```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.3
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - esdata:/usr/share/elasticsearch/data
    restart: unless-stopped

volumes:
  esdata:
    driver: local
  qdrant_data:
    driver: local
```

## Data Indexing

The included `index.py` script processes and indexes the arXiv dataset (~1.7 million papers) into Elasticsearch:

```python
# Key statistics from indexing:
# - Total documents indexed: 1.7 million
# - Time taken: ~675 seconds
# - Average speed: ~3980 documents/second
```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Enter a research topic in the search box
   - Click "Search Papers" to find relevant papers
   - Select papers to download and process
   - After processing, ask questions about the papers in the chat interface

## Configuration Options

- **Model Selection**: Choose from various Ollama models for different performance characteristics
- **Maximum Papers**: Set the maximum number of papers to fetch (1-20)
- **Conversation Management**: Clear conversation history or reset the knowledge base as needed

## Project Structure

- `app.py`: Main Streamlit application
- `paper_process.py`: Functions for fetching, downloading, and processing papers
- `chat_chain.py`: Creates the chat chain for QA functionality
- `search.py`: Handles Elasticsearch queries and paper discovery
- `docker-compose.yml`: Docker configuration for Elasticsearch
- `index_arxiv.py`: Script for indexing arXiv dataset into Elasticsearch

## How It Works

1. **Search**: Uses Elasticsearch to find papers matching your query
2. **Download**: Selected papers are downloaded and processed
3. **Indexing**: Papers are split into chunks and embedded using sentence transformers
4. **Storage**: Embeddings are stored in Qdrant vector database
5. **Retrieval**: When you ask a question, the system finds relevant text chunks
6. **Response**: The selected LLM generates an answer based on the retrieved content

## Dataset

This system uses the arXiv dataset available on Kaggle:
- Source: https://www.kaggle.com/datasets/Cornell-University/arxiv
- Size: ~1.7 million academic papers
- Format: JSON Lines (.jsonl)
- Content: Paper metadata including titles, abstracts, authors, categories, and more

## Performance Notes

- Elasticsearch indexing performance: ~4000 documents/second on standard hardware
- Full dataset indexing takes approximately 11-12 minutes
- Paper processing time depends on PDF size and complexity
- Response time varies based on the Ollama model selected

## Limitations

- Can only process papers available on arXiv
- PDF processing quality depends on the structure of the original document
- Response quality depends on the selected Ollama model
- Requires sufficient memory for Elasticsearch (minimum 512MB heap allocated)

## Future Improvements

- Support for additional academic sources beyond arXiv
- Improved PDF processing for complex layouts
- Citation integration for answers
- User profiles and saved research topics
- Cloud deployment options
- Parallel processing of multiple papers

## Acknowledgments

- [arXiv](https://arxiv.org/) for providing open access to research papers
- [LangChain](https://github.com/langchain-ai/langchain) for document processing and LLM integration
- [Streamlit](https://streamlit.io/) for the web interface
- [Qdrant](https://qdrant.tech/) for vector storage
- [HuggingFace](https://huggingface.co/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM support
- [Elasticsearch](https://www.elastic.co/) for search functionality