import os
from langchain_ollama import ChatOllama

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOllama(model = "mistral")

question = input("Enter a question: ")
response = llm.invoke(question)
print(response.content)