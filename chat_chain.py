from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def create_chat_chain(retriever, model_name):
    """Create a conversational chain that maintains chat history"""
    llm = OllamaLLM(model=model_name)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create a custom prompt template that instructs the model to be accurate and stay within context
    prompt_template = """
        You are an AI research assistant specializing in summarizing and explaining research papers. 
        Your task is to answer user questions strictly based on the provided context. 

        **Instructions:**
        - Use only the given context to generate responses.
        - Do not provide information outside the context, even if relevant.
        - If the context does not contain enough information, state: "The provided context does not include details on this topic."
        - Provide clear, structured, and well-explained answers.

        ### Context:
        {context}

        ### Chat History:
        {chat_history}

        ### User's Question:
        {question}

        ### Detailed Answer:

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