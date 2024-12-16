import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Agile Coach. Answer any questions"
         "related to the agile process"),
        ("human", "{question}")
    ]
)

st.title("Agile Guide")

question = st.text_input("Enter the question: ")

chain = prompt_template | llm

if question:
    response = chain.invoke({
        "question":question
    })
    st.write(response.content)