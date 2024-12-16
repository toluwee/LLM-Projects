import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = FAISS.from_documents(chunks, llm)
retriever = db.as_retriever()
text = input("Enter a query")

# embedding_vector = llm.embed_query(text)
# docs = db.similarity_search_by_vector(embedding_vector)

docs = retriever.invoke(text)
for doc in docs:
    print(doc.page_content)