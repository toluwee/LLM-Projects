import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Agile Coach. Answer any questions"
         "related to the agile process"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

chain = prompt_template | llm

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id:history_for_chain,
    input_messages_key="question",
    history_messages_key="chat_history"
)

st.title("Agile Guide")

question = st.text_input("Enter the question: ")

if question:
    response = chain_with_history.invoke({
        "question":question},
        {"configurable":{"session_id":"abc123"}
    })
    st.write(response.content)

st.write("HISTORY")
st.write(history_for_chain)