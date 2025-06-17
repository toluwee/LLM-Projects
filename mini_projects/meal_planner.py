import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

# Define the prompt template
prompt_template = """
You are a meal planning assistant. 
Create a meal plan for {num_days} days based on the following 
dietary restrictions: {dietary_restrictions} and 
daily caloric requirement: {caloric_requirement}.
Include healthy snack options daily such as apple with peanut butter etc.
For each meal, include a link to the recipe.
Provide a weekly shopping list based on the recipes for that week
"""

# Create the prompt
prompt = PromptTemplate(
    input_variables=["num_days", "dietary_restrictions", "caloric_requirement"],
    template=prompt_template
)

# Initialize the language model
llm = ChatOpenAI(model=llm_model, temperature=0.0, api_key=OPENAI_API_KEY)

# Create the RunnableSequence
llm_chain = prompt | llm

# Define the function to create a meal plan
def create_meal_plan(num_days, dietary_restrictions, caloric_requirement):
    response = llm_chain.invoke({
        "num_days": num_days,
        "dietary_restrictions": dietary_restrictions,
        "caloric_requirement": caloric_requirement
    })
    return response.content

# Streamlit application
st.title("Meal Plan Generator")

# User inputs
days = st.number_input("Number of days", min_value=1, max_value=30, value=7)
diet_restrictions = st.text_input("Dietary restrictions (e.g., vegetarian, gluten-free)")
calorie_requirement = st.number_input("Daily caloric requirement (in kcal)", min_value=1000, max_value=4000, value=2000)

# Generate meal plan button
if st.button("Generate Meal Plan"):
    if diet_restrictions and calorie_requirement:
        meal_plan = create_meal_plan(days, diet_restrictions, calorie_requirement)
        st.write(meal_plan)
    else:
        st.error("Please enter dietary restrictions and caloric requirement.")