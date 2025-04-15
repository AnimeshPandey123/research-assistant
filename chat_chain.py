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
        - Never use accronyms or abbreviations unless they are widely recognized.
        - You can however use context to get to a conclusion that is not directly stated in the context.
        - The context is mainly scientific papers.
        - Do not provide information outside the context, even if relevant.
        - If the context does not contain enough information, state: "The provided papers does not include details on this topic. I can only answer based on the information provided. Please ask me about the papers."
        - Provide clear, structured, and well-explained answers.
        - Be very descriptive about the answer and give full explanation.

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
        verbose=True,
    )

    return chain