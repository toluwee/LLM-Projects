import os
from langchain_openai import ChatOpenAI
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm=ChatOpenAI(
    model=  "deepseek-chat", 
    api_key=DEEPSEEK_API_KEY,
    temperature=0.0,
    max_tokens=1000,
    base_url="https://api.deepseek.com"
    )

prompt = hub.pull("hwchase17/react")

tools = load_tools([
    "wikipedia",
    "ddg-search",
    ])

agent= create_react_agent(llm,tools,prompt)

agent_executor=AgentExecutor(
    agent= agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    )

st.title("AI Agent")
task=st.text_input("Assign me a task")

if task:
    try:
        response = agent_executor.invoke({"input": task})
        st.write(response["output"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try again in a few moments.")







