import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def load_documents(folder_path):
    """Load documents from multiple file formats"""
    documents = []
    supported_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
    UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension in supported_extensions:
            try:
                loader = UnstructuredLoader(file_path,
                            api_key=UNSTRUCTURED_API_KEY,
                            partition_via_api=True,
                            chunking_strategy="by_title",
                            strategy="fast")
                documents.extend(loader.load())
                print(f"Successfully loaded: {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")

    return documents


def setup_qa_chain():
    """Set up the QA chain with embeddings and LLM"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("Please set your OPENAI_API_KEY environment variable")
        st.stop()

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

    return embeddings, llm


def create_qa_system(documents, embeddings, llm):
    """Create the QA system with vector store and retrieval chain"""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()

    # Set up prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for answering questions.
                    Use the provided context to respond. If the answer 
                    isn't clear, acknowledge that you don't know. 
                    Limit your response to three concise sentences.
                    {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create retrieval chain
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)
    qa_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain


def main():
    st.title("Multi-format Document QA System")

    # Input for folder path
    folder_path = st.text_input(
        "Enter folder path containing documents:",
        placeholder="e.g., C:/Documents/my_files"
    )

    if not folder_path:
        st.warning("Please enter a folder path")
        return

    if not os.path.exists(folder_path):
        st.error("Folder path does not exist")
        return

    # Load documents and set up QA system
    documents = load_documents(folder_path)

    if not documents:
        st.error("No supported documents found in the specified folder")
        return

    embeddings, llm = setup_qa_chain()
    rag_chain = create_qa_system(documents, embeddings, llm)

    # Set up chat history
    history_for_chain = StreamlitChatMessageHistory()
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # Question input
    question = st.text_input("Ask your question:")

    if question:
        with st.spinner("Processing your question..."):
            response = chain_with_history.invoke(
                {"input": question},
                {"configurable": {"session_id": "abc123"}}
            )
            st.write(response.get('answer', 'No answer found'))


if __name__ == "__main__":
    main()