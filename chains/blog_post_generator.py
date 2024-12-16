import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)

# Define the PromptTemplates
outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
    - Introduction
    - 3 main points with sub-points
    - Conclusion
    """
)

introduction_prompt = PromptTemplate(
    input_variables=["outline", "number_of_paragraphs"],
    template="""
    You are a professional blogger.
    Write an engaging introduction paragraph based on the following
    outline:{outline}
    The introduction should hook the reader and provide a brief
    overview of the topic with {number_of_paragraphs} paragraphs.
    """
)

# Define the chains
first_chain = outline_prompt | llm | StrOutputParser() | (lambda outline: (st.write(outline), outline)[1])
second_chain = introduction_prompt | llm

# Streamlit UI
st.title("Speech Generator")

topic = st.text_input("Enter a topic: ")
number_of_paragraphs = st.number_input("Enter a number of paragraphs: ", min_value=1, max_value=10)

if topic and number_of_paragraphs:
    # First, get the outline from the first chain
    outline = first_chain.invoke({"topic": topic})

    # Then, use the title to get the speech from the second chain
    response = second_chain.invoke({
        "number_of_paragraphs": number_of_paragraphs,
        "outline": outline
    })

    st.write(response.content)