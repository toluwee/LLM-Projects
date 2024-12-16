import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
# from langchain.globals import set_debug

# set_debug(True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = "gpt-4o", api_key=OPENAI_API_KEY)

prompt_template = PromptTemplate(
    input_variable=["city", "month", "language", "budget"],
    template = """
    Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!

    """

)

st.title("Travel Guide")

city = st.text_input("Enter a city: ")
month = st.text_input("Enter a month of travel: ")
language = st.text_input("Enter a language: ")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

if city and month and language and budget:
    response = llm.invoke(prompt_template.format(
        city=city, month=month, language=language, budget=budget

    ))
    st.write(response.content)