import chromadb
from chromadb import PersistentClient
from llama_index.core import (
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import os

llm = None


Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model


persist_dir = r"C:\Users\kiran\myOwnRag\chroma_db"
os.makedirs(persist_dir, exist_ok=True)
chroma_client = PersistentClient(path=persist_dir)
chroma_collection = chroma_client.get_or_create_collection("mydefensecollection")


vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)


custom_prompt = PromptTemplate(
    "You are a military analyst. Use the context below to answer the user's question clearly and concisely.\n"
    "If the answer is not present in the documents, reply with 'Not found in the documents.'\n\n"
    "Context:\n{context_str}\n\n"
    "Question:\n{query_str}\n\n"
    "Answer:"
)


query_engine = index.as_query_engine(
    llm=Settings.llm,
    text_qa_template=custom_prompt,
    similarity_top_k=3
)


query = input("Enter your question: ")
response = query_engine.query(query)
print(response)
