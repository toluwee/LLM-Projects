import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

llm = ChatOpenAI(model = "mistral")

prompt_template = PromptTemplate(
    input_variable=["company", "position", "strengths", "weaknesses"],
    template = """
    You are a career coach who, given the {company}, {position}, 
    {strengths}, and {weaknesses}, provides personalized 
    interview tips to prepare for success. Give short motivational insight at end of tip

    """

)

st.title("Interview Tips Generator")

company = st.text_input("Company Name")
position = st.text_input("Position Title")
strengths = st.text_area("Your Strengths", placeholder="Write some of your strengths", max_chars=100, height= 5)
weaknesses = st.text_area("Your Weaknesses", placeholder="Write some of your weaknesses", max_chars=100, height= 5)

if company and position and strengths and weaknesses:
    response = llm.invoke(prompt_template.format(
        company = company,
        position = position,
        strengths = strengths,
        weaknesses = weaknesses
    ))
    st.write(response.content)