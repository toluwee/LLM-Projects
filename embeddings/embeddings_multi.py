import os
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

response = llm.embed_documents(
    [
        "I love playing video games",
        "I am going to the movie",
        "I love coding",
        "Hello World!"
    ]
)
print(len(response))
print(response[2])