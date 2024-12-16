import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import datetime

# account for deprecation of LLM model
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 30)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-4o-mini"
else:
    llm_model = "gpt-4o-mini-2024-07-18"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model = llm_model, temperature=0.0, api_key=OPENAI_API_KEY)

# Define the PromptTemplates
product_prompt = PromptTemplate(
    input_variables=["product_name","features"],
    template="""
    You are an experienced marketing specialist. 
    Create a catchy subject line for a marketing 
    email promoting the following product: {product_name}. 
    Highlight these features: {features}. 
    Respond with only the subject line.
    """
)

email_prompt = PromptTemplate(
    input_variables=["product_name", "subject_line", "target_audience"],
    template="""
    Write a marketing email of 300 words for the 
    product: {product_name}. Use the subject line:
    {subject_line}. Tailor the message for the 
    following target audience: {target_audience}.
    Format the output as a JSON object with three 
    keys: 'subject', 'audience', 'email' and fill 
    them with respective values.
    """
)

# Define the chains
first_chain = product_prompt | llm | StrOutputParser()
second_chain = email_prompt | llm |JsonOutputParser()
final_chain = first_chain | (lambda subject_line: {"product_name": product_name,
                                            "subject_line": subject_line,
                                            "target_audience": target_audience}
                             ) | second_chain

# Streamlit UI
st.title("Marketing Email Generator")

product_name = st.text_input("Enter a product name: ")
features = st.text_input("State features of the product: ")
target_audience = st.text_input("Enter a target audience: ")

if product_name and features and target_audience:
    response = final_chain.invoke({"product_name": product_name,
                                   "features": features})
    st.write(response)