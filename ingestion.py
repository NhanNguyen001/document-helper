import os
import certifi
import codecs

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain_core.documents import Document



class UTF8ReadTheDocsLoader(ReadTheDocsLoader):
    """Custom loader that ensures UTF-8 encoding throughout the loading process"""
    
    def __init__(self, path: str):
        """Initialize with path."""
        super().__init__(path)
        self.file_paths = []
        # Walk through the directory and collect HTML files
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.html'):
                    file_path = os.path.join(dirpath, filename)
                    self.file_paths.append(file_path)
    
    def _read_file(self, file_path: str) -> str:
        try:
            # First try UTF-8
            with codecs.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except UnicodeError:
            # Fallback to cp1252 (Windows default) with error handling
            with codecs.open(file_path, 'r', encoding='cp1252', errors='replace') as f:
                return f.read()

    def lazy_load(self):
        """Lazy load the documents."""
        from bs4 import BeautifulSoup
        
        for file_path in self.file_paths:
            try:
                content = self._read_file(file_path)
                soup = BeautifulSoup(content, "html.parser")
                # Remove all script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                # Handle newlines and excessive whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                metadata = {"source": file_path}
                yield Document(page_content=text, metadata=metadata)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def process_batch(documents: List[Document], batch_size: int = 50) -> None:
    """Process documents in batches and add them to Pinecone."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}, documents {i} to {i + len(batch)}"
        )
        PineconeVectorStore.from_documents(
            batch, embeddings, index_name="langchain-doc-index"
        )


def ingest_docs():
    loader = UTF8ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest/")
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
