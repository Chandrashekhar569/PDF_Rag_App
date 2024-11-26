import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')


llm = ChatGroq(groq_api_key=groq_api_key,model='gemma-7b-it')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}

    """
)

# Streamlit
st.set_page_config(page_title="PDF Rag_APP", page_icon=":books")
st.title("DocuSift: Smart PDF Query Tool")

def save_uploaded_file(uploaded_file):
    save_dir = 'uploaded_files'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the file in the directory
    with open(os.path.join(save_dir, uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return st.sidebar.success(f'File saved at {os.path.join(save_dir, uploaded_file.name)}')

# Create a file uploader
st.sidebar.title('Upload a PDF file')

# Allow user to upload a PDF file
uploaded_file = st.sidebar.file_uploader('Choose a PDF file', type='pdf')

# Save the file if it is uploaded
if uploaded_file is not None:
    save_uploaded_file(uploaded_file)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("uploaded_files")
        st.session_state.documents=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


if st.sidebar.button("Document Embending"):
    create_vector_embeddings()
    st.sidebar.success("Vector Database created successfully. Now you Can ask questions.")

user_prompt = st.text_input("Enter youur query from research paper:")

search_button = st.button("Search")
if search_button:
    document_cahin = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_cahin)
    response = retriever_chain.invoke({"input":user_prompt})
    st.write(response['answer'])
