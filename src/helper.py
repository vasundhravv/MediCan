from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HugfaceEmbeddings
from typing import List
from langchain.schema import Document


# load pdf files to the directory
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
        )
    documents = loader.load()
    return documents
