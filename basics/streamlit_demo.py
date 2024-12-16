import os
from langchain_openai import ChatOpenAI
import streamlit as st
# from langchain.globals import set_debug

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)

st.title("What do you want to Know?")

question = st.text_input("Enter a question: ")

if question:
    response = llm.invoke(question)
    st.write(response.content)