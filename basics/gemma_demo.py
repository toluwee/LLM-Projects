import os
from langchain_ollama import ChatOllama

llm = ChatOllama(model = "gemma:2b")

question = input("Enter a question: ")
response = llm.invoke(question)
print(response.content)

