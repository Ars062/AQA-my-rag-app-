import os
import tempfile
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import ollama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader
from typing import List
import chromadb
import string
import requests
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
system_prompt = """You are an intelligent assistant designed to answer user queries **only using the information provided in the uploaded documents, URLs, or input text**. Do not use any external knowledge, assumptions, or hallucinations.

Your responses must:
- Be **factually grounded** in the provided content only
- Use **clear and concise language**
- Follow a **logical flow**
- Be **well-structured** for readability:
  - Use **paragraphs** to explain key points
  - Include **bullet points** or **numbered lists** where appropriate
  - Highlight key facts clearly
- If a direct answer is not found in the documents, **politely state that** the required information was not available

Do not fabricate or guess. Always stick strictly to the contents of the user-provided materials.
If you do not have enough information to answer the question, please say so politely and suggest that the user provide more context or details."""

class OllamaEmbeddingFunction(DefaultEmbeddingFunction):
    def __init__(self, model="nomic-embed-text", url="http://localhost:11434/api/embeddings"):
        self.model = model
        self.url = url

    def __call__(self, texts):
        embeddings = []
        for text in texts:
            response = requests.post(self.url, json={"model": self.model, "prompt": text})
            if response.status_code == 200:
                embeddings.append(response.json()["embedding"])
            else:
                raise Exception(f"Failed to get embedding from Ollama: {response.text}")
        return embeddings


st.set_page_config(
        page_title="AQA Search",
        page_icon=":mag_right:",
        layout="wide",
        initial_sidebar_state="expanded")
st.header("AQA Search")
def get_embedding_function()->chromadb.Collection:
    ollama_ef= OllamaEmbeddingFunction(
    model="nomic-embed-text:latest",
    url="http://localhost:11434/api/embedding"
    )
    chromadb_client = chromadb.PersistentClient(path="./chroma_db" )
    return chromadb_client.get_or_create_collection(
                                             
        embedding_function=ollama_ef,
        name="aqa-search",
        metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:M": 16,}

    )
def add_to_vector_store(all_splits: List[Document], file_name: str = None):
    collection = get_embedding_function()
    
    # Delete old chunks with the same source file name
    if file_name:
        old_items = collection.get(where={"source": file_name})
        if old_items and "ids" in old_items and old_items["ids"]:
            collection.delete(ids=old_items["ids"])
            st.info(f"Removed old chunks for {file_name} from the vector store.")
    
    # Add new chunks
    documents, metadata, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadata.append({"source": file_name, "id": idx})
        ids.append(f"{file_name}_{idx}")
    
    collection.upsert(
        documents=documents,
        metadatas=metadata,
        ids=ids
    )
    st.success(f"Added {len(all_splits)} chunks from {file_name} to the vector store.")




def process_inputs(uploaded_files: List[UploadedFile] = None, urls: List[str] = None, user_inputs: List[str] = None) -> List[Document]:
    documents = []

    # File loading
    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=file.name) as temp_file:
                temp_file.write(file.getvalue())
                file_path = temp_file.name

            if file.type == "application/pdf":
                loader = PyPDFLoader(file_path)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file.type == "text/plain":
                loader = TextLoader(file_path)
            elif file.type in ["text/html", "text/x-html"]:
                loader = UnstructuredHTMLLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file.type}")
                os.unlink(file_path)
                continue

            documents.extend(loader.load())
            os.unlink(file_path)

    # Text and URLs as documents
    if user_inputs:
        for text in user_inputs:
            documents.append(Document(page_content=text))

    if urls:
        for url in urls:
            documents.append(Document(page_content=f"Placeholder content from: {url}"))

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n", "\n\n", " ", ",", "?"]
    )
    return text_splitter.split_documents(documents)

def query_vector_store(prompt: str, n_results: int = 10):
    collection = get_embedding_function()
    results = collection.query(
        query_texts=[prompt],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(documents, metadatas)
    ]

def call_llm(prompt: str, context: List[Document]):
    context_text = "\n\n".join([doc.page_content for doc in context])
    response = ollama.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"context:\n{context_text}\n\nQuestion: {prompt}"},
        ],
        model="llama3.1:latest",
        stream=True,
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]
        else:
            break


    


def re_rank_cross_encoder(prompt: str, documents: List[str]) -> tuple[str, list[int]]:
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder = CrossEncoder(model_name)

    pairs = [[prompt, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]

    relevant_text = "\n\n".join([documents[idx] for idx, _ in ranked])
    relevant_text_ids = [idx for idx, _ in ranked]
    return relevant_text, relevant_text_ids

if __name__ == "__main__":
 with st.sidebar:
    st.title("AQA Search")
    st.write("Upload your files or enter URLs for question answering.")
    uploaded_file = st.file_uploader("Upload a PDF or DOCX or url for question answer", type=["pdf", "docx", "html", "htm", "txt"],accept_multiple_files=True , )
    if uploaded_file is not None:
        for file in uploaded_file:
            st.write(f"Uploaded file: {file.name}")    
     # URL input
        urls = st.text_area("Enter URLs (one per line):", height=100)

        # Direct text input
        user_input = st.text_area("Enter text:", height=100)
    
    if uploaded_file:
        button_label = "📂 Process Files"
    elif urls.strip():
        button_label = "🌐 Process URLs"
    elif user_input.strip():
        button_label = "📝 Process Text"
    else:
        button_label = "⚪ Awaiting Input..."

    process = st.button(button_label)
    if process:
        if uploaded_file:
            all_splitted_texts = process_inputs(uploaded_files=uploaded_file)
            st.write(all_splitted_texts)
        elif urls.strip():
            all_splitted_texts = process_inputs(urls=urls.splitlines())
            st.write(all_splitted_texts)
        elif user_input.strip():
            all_splitted_texts = process_inputs(user_inputs=[user_input])
            st.write(all_splitted_texts)
        else:
            st.warning("Please upload files, enter URLs, or provide text input.")

    if process and (uploaded_file or urls or user_input):
         if urls:
            normalized_urls = [url.strip() for url in urls.splitlines() if url.strip()]
            if not normalized_urls:
             urls = None 
         elif user_input:
            normalized_user_input = user_input.strip()
            if not normalized_user_input:
                user_input = None
         elif uploaded_file:
            first_file_name = uploaded_file[0].name
            normalized_uploaded_file = first_file_name.translate(str.maketrans("", "", string.punctuation)).replace(" ", "_")  
         else:
            normalized_uploaded_file = None
         all_splitted_texts = process_inputs(
            uploaded_files=uploaded_file,
            urls=urls.splitlines() if urls else None,
            user_inputs=[user_input] if user_input else None
        )
         add_to_vector_store(all_splits=all_splitted_texts, file_name=normalized_uploaded_file)

         st.success("Data processed and added to vector store.")
st.header("Ask a Question")
prompt = st.text_input("Enter your question:")
ask= st.button("Ask")
if ask and prompt:

    results = query_vector_store(prompt)
    if results:
        st.write("Results:")
        for result in results:
            st.write(f"Source: {result.metadata.get('source', 'Unknown')}")
            st.write(result.page_content)
            st.write("---")
    else:
        st.warning("No results found for your question.")
    context = results
    relevant_text, relevant_text_ids = re_rank_cross_encoder(prompt, [doc.page_content for doc in context])
    response= call_llm(prompt=prompt, context=context)
    st.write_stream(response)
    with st.expander("Re-ranked Context"):
        st.write(relevant_text)
        st.write("Relevant IDs:", relevant_text_ids)
    with st.expander("LLM Response"):
     st.write("Answer (above was streamed):")
