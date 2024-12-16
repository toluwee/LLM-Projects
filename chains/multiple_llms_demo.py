import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm1 = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)
llm2 = ChatOllama(model = "mistral")

# Define the PromptTemplates
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title. 
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title", "number_of_paragraphs", "language"],
    template="""
    You need to write a powerful speech of 350 words 
    with {number_of_paragraphs} paragraphs in {language} language
    for the following title: {title} 
    """
)

# Define the chains
first_chain = title_prompt | llm1 | StrOutputParser() | (lambda title: (st.write(title), title)[1])
second_chain = speech_prompt | llm2
final_chain = first_chain | second_chain

# Streamlit UI
st.title("Speech Generator")

topic = st.text_input("Enter a topic: ")
number_of_paragraphs = st.number_input("Enter a number of paragraphs: ", min_value=1, max_value=7)
language = st.text_input("Enter a language: ")

if topic and number_of_paragraphs and language:
    # First, get the title from the first chain
    title = first_chain.invoke({"topic": topic})

    # Then, use the title to get the speech from the second chain
    response = second_chain.invoke({
        "title": title,
        "number_of_paragraphs": number_of_paragraphs,
        "language": language
    })

    st.write(response.content)