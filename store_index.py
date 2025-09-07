import os 
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY


extracted_data = load_pdf_files(data="data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embeddings = download_hugging_face_embeddings()


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

index_name = "medican"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, #demension of the embedding model
        metric="cosine", #cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)

    # load existing index


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)