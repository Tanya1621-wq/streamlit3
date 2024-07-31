import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)  # Ensure the file pointer is at the start
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(txt):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(txt)
    return chunks

def get_vect_store(chunks):
    embd = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embd)
    vector_store.save_local("faiss_index")

def user_input(ques):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(ques)

    llm = ChatOllama(model="llama3")
    prompt_template = """
    You are a professional who is expert in reading and analyzing PDFs. Answer the question as detailed as possible from the provided context. If the information is not in the provided context, just say, "Answer not in the context." Do not give a wrong answer.
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    context = " ".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "question": ques})

    # Check the structure of response and print it correctly
    if isinstance(response, str):
        st.write("Reply:", response)
    elif isinstance(response, dict):
        st.write("Reply:", response.get("output", "No output key in response"))
    else:
        st.write("Unexpected response format")

def main():
    st.title("Document-based Chatbot")
    user_question = st.text_input("How may I help you?")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files here", type="pdf", accept_multiple_files=True)
        
        if pdf_docs and st.button("Submit"):
            with st.spinner("Analyzing..."):
                raw_text = get_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vect_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
