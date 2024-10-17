import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.schema import Document  
from dotenv import load_dotenv


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
api_key = os.getenv('GROQ_API_KEY')


llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")


prompt = ChatPromptTemplate.from_template(
    """Answer the questions based only on the provided context.
    Please provide the most accurate answer.
    <context>
    {context}
    <context>
    Question: {input}
    """
)


def create_vector_embeddings(uploaded_file):
    if "vectorstore" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        all_text = ""
        pdf_reader = PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            all_text += page.extract_text()
        
        document = Document(page_content=all_text)
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents([document])
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("RAG Document Q&A With Groq and HuggingFace")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

user_prompt = st.text_input("Ask a query from the uploaded document")

if st.button("Process Document") and uploaded_file:
    create_vector_embeddings(uploaded_file)
    st.success("Vector Database ready")

if user_prompt and "vectorstore" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': user_prompt})

    if 'answer' in response:
        st.write(response['answer'])
    if 'context' in response:
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("------------------------------")
