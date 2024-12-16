import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

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
    input_variables=["title", "emotion", "number_of_paragraphs"],
    template="""
    You need to write a powerful {emotion} speech of 350 words 
    with {number_of_paragraphs} paragraphs
    for the following title: {title} 
    Format the output with 3 keys: 'title', 'speech', 'emotion' and fill 
    them with the respective values
    """
)

# Define the chains
first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title), title)[
    1])  #the lambda helps to print the title on the browser while the StrOutputParser()
## gets the title to be used in the second chain
second_chain = speech_prompt | llm | JsonOutputParser()
final_chain = first_chain | (lambda title: {"title": title,
                                            "emotion": emotion,
                                            "number_of_paragraphs": number_of_paragraphs}
                             ) | second_chain

# Streamlit UI
st.title("Speech Generator")

topic = st.text_input("Enter a topic: ")
emotion = st.text_input("Enter an emotion: ")
number_of_paragraphs = st.number_input("Enter a number of paragraphs: ", min_value=1, max_value=7)

if topic and number_of_paragraphs and emotion:
    # First, get the title from the first chain
    response = final_chain.invoke({"topic": topic})
    st.write(response)
