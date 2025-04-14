import chromadb
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

llm = None

Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("mydefensetest")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(r"C:\Users\kiran\myOwnRag\data").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

query_engine = index.as_query_engine(llm=Settings.llm)
response = query_engine.query("Summarize India's defense industrial strategy.")
print(response)
