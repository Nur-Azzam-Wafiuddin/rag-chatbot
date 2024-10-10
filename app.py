import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq



import os
from io import StringIO
from chromadb.config import Settings

# Initialize the embedding model
embed = OllamaEmbeddings(model="nomic-embed-text") # Nomic Embed
# embed = OllamaEmbeddings(model="llama3") # Llama 3

st.set_page_config(layout="wide")
st.title("Document Chatbot")

# Initialize the language model
# llm = OllamaLLM(model="llama3") # llama3
llm = ChatGroq(
    temperature=0,
    groq_api_key = os.getenv("GROQ_API"),
    model_name="llama-3.1-70b-versatile"
) # Groq

st.title("Upload Document")
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize session state for storing embeddings
if "db" not in st.session_state:
    st.session_state.db = None
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

# Check if new files are uploaded
if uploaded_files:
    new_files = False
    documents = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_file_names:
            new_files = True
            st.session_state.uploaded_file_names.append(uploaded_file.name)
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Load PDF from the saved temp file
            loader = PyPDFLoader(f"temp_{uploaded_file.name}")
            documents.extend(loader.load())
    
    if new_files:
        st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Create the document embedding database using in-memory FAISS
        db = FAISS.from_documents(texts, embed, docstore=InMemoryDocstore(), index_to_docstore_id={})
        st.session_state.db = db  # Save to session state
    else:
        st.info("No new documents to upload.")

# If there are stored embeddings, create a retriever for searching documents
if st.session_state.db:
    retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    st.title("Simple Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        relevant_docs = retriever.invoke(prompt)
        
        # Combine the query and the relevant document contents
        combined_input = (
            "Here are some documents that might help answer the question: "
            + prompt
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
        )
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ]
        print(messages)
        result = llm.invoke(messages)

        with st.chat_message("assistant"):
            response = result.content  # Groq API
            # response = result          # Ollama
            st.markdown(response)
            for i, doc in enumerate(relevant_docs, 1):
                with st.expander(f"Document {i} Context"):
                    st.write(doc.metadata)
                    st.write(doc.page_content)
                    print(doc.metadata)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

