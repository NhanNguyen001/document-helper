from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

load_dotenv()


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

    docsearch = PineconeVectorStore(
        index_name="langchain-doc-index", embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain,
    )

    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm("What is a LangChain?")
    print(res["answer"])
