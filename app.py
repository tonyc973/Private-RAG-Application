import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(
    page_title="Local PDF Q&A ✨",
    page_icon="📄",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>💡 Ask your PDF — Locally, Privately</h1>", unsafe_allow_html=True)
st.markdown("#### Powered by [Mistral 7B GGUF + LangChain + FAISS + llama.cpp]")

uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Reading and chunking your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name


        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        llm = LlamaCpp(
            model_path="/home/george/llama_models/llama.cpp/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.3,
            max_tokens=512,
            n_ctx=10000,
            n_threads=16,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            streaming=True,
            verbose=False
        )

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDF processed. Ask your question below.")
        user_query = st.text_input("❓ Ask a question about the PDF")

        if user_query:
            with st.spinner("💬 Thinking..."):
                answer = qa_chain.run(user_query)
            st.markdown("### ✅ Answer:")
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>{answer}</div>", unsafe_allow_html=True)

st.markdown("---")

