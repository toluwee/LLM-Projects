# import os
# from langchain_ollama import ChatOllama
#
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOllama(model = "gemma:2b")
#
# question = input("Enter a question: ")
# response = llm.invoke(question)
# print(response.content)

import inspect
from langchain_community.agent_toolkits import load_tools

# Print the source code of load_tools to see the supported tools
print(inspect.getsource(load_tools))