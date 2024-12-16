import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
# from langchain.globals import set_debug

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)

prompt_template = PromptTemplate(
    input_variable=["country"],
    template = """
    You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {number_of_paragraphs} short paras in {language}

    """

)

st.title("Cuisine Guru")

country = st.text_input("Enter a country: ")
number_of_paragraphs = st.number_input("Enter a number of paragraphs: ", min_value=1, max_value=7)
language = st.text_input("Enter a language: ")

if country and number_of_paragraphs and language:
    response = llm.invoke(prompt_template.format(country=country, number_of_paragraphs=number_of_paragraphs, language=language))
    st.write(response.content)