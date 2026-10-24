import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

st.title("📘 RAG System - 48 Laws of Power")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    pdf_path = os.path.join("temp.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Try PyPDFLoader
    data = PyPDFLoader(pdf_path)
    docs = data.load()

    # Fallback to OCR if text is too small
    if sum(len(d.page_content.strip()) for d in docs) < 1000:
        st.info("PDF seems image-based, running OCR...")
        pages = convert_from_path(pdf_path, first_page=1, last_page=50)  # limit for speed
        text = "".join(pytesseract.image_to_string(page) for page in pages)
        docs = [Document(page_content=text)]
        st.write("OCR preview:", text[:300])

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings + Chroma
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="chroma_db")

    st.success(f"Stored {len(chunks)} chunks in Chroma DB.")

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.1})
    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context if relevant, otherwise answer from general knowledge."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    query = st.text_input("Enter your question:")
    if query:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:**\n{doc.page_content[:500]}...")
