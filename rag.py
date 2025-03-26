import os
import time
import requests
from tqdm.notebook import tqdm
from typing import Optional, List, Tuple
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer

# from transformers import AutoTokenizer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy



from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from paper_process import fetch_papers

import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

import pacmap
import numpy as np
import plotly.express as px


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]
def process_pdfs(pdf_paths: List[str]) -> List[LangchainDocument]:
    print("Processing PDFs...")
    docs = []
    for pdf_path in tqdm(pdf_paths, desc="Loading PDFs"):
        loader = PyPDFLoader(pdf_path['path'])
        pages = loader.load()
        for page in pages:
            docs.append(LangchainDocument(page_content=page.page_content, metadata={"source": pdf_path}))
    
    return split_documents(512, docs, tokenizer_name=EMBEDDING_MODEL_NAME)


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


import multiprocessing
if __name__ == '__main__':
    # # Set the start method explicitly
    # multiprocessing.set_start_method('spawn')
    
    # # Disable tokenizer parallelism
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Example usage
    query = "machine learning"
    dirpath, pdf_paths = fetch_papers(query)
    docs_processed = process_pdfs(pdf_paths)



    # To get the value of the max sequence_length, we will query the underlying `SentenceTransformer` object used in the RecursiveCharacterTextSplitter
    print(
        f"Model's maximum sequence length: {SentenceTransformer('thenlper/gte-small').max_seq_length}"
    )


    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

    # Plot the distribution of document lengths, counted as the number of tokens
    # fig = pd.Series(lengths).hist()
    # plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    # plt.show()



    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        # model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    # Embed a user query in the same space
    user_query = "What are the conclusions?"
    query_vector = embedding_model.embed_query(user_query)
    print(query_vector)


    embedding_projector = pacmap.PaCMAP(
        n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
    )

    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0])
        for idx in range(len(docs_processed))
    ] + [query_vector]

    # Fit the data (the index of transformed data corresponds to the index of the original data)
    documents_projected = embedding_projector.fit_transform(
        np.array(embeddings_2d), init="pca"
    )

    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    print(
        "\n==================================Top document=================================="
    )
    print(retrieved_docs[0].page_content)
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)


    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     READER_MODEL_NAME, quantization_config=bnb_config
    # )
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        device="cpu",
        max_new_tokens=300,
    )

    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    print(RAG_PROMPT_TEMPLATE)

    retrieved_docs_text = [
        doc.page_content for doc in retrieved_docs
    ]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    print(context)

    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question="What are the conclusions?", context=context
    )

    # Redact an answer
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    print(answer)

