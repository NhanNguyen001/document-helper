from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def process_batch(documents: List[Document], batch_size: int = 50) -> None:
    """Process documents in batches and add them to Pinecone."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}, documents {i} to {i + len(batch)}")
        PineconeVectorStore.from_documents(
            batch, embeddings, index_name="langchain-doc-index"
        )

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest/")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_ur = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_ur})

    print(f"Going to add {len(documents)} to Pinecone")
    
    # Process documents in batches
    process_batch(documents)
    
    print("Finished processing all documents")

if __name__ == "__main__":
    ingest_docs()
