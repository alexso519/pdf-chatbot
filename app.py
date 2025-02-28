import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from transformers import pipeline
import os
import uuid
from datetime import datetime

# Set Hugging Face endpoint
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# Custom CSS for modern UI
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F6;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #1E90FF !important;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0077B6;
    }
    .stFileUploader>div>div>div>button {
        background-color: #FF6347;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stSuccess {
        color: #32CD32 !important;
    }
    .stError {
        color: #FF6347 !important;
    }
    .stExpander {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        max-height: 80vh;
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for mode selection
st.sidebar.title("Mode")
mode = st.sidebar.selectbox("Choose mode", ["Owner Portal", "Chatbot"])

# Shared vector store initialization
def load_vector_store():
    if os.path.exists("vector_store"):
        embedder = HuggingFaceEmbeddings()
        return FAISS.load_local(
            "vector_store",
            embedder,
            allow_dangerous_deserialization=True
        )
    return None

# Owner Portal Functions
def list_uploaded_pdfs():
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    return [f for f in os.listdir("uploads") if f.endswith(".pdf")]

def delete_pdf(filename):
    if os.path.exists(f"uploads/{filename}"):
        os.remove(f"uploads/{filename}")
        st.success(f"Deleted: {filename}")
    else:
        st.error(f"File not found: {filename}")

def process_and_add_to_vector_store(filename):
    try:
        with st.spinner(f"Processing {filename}..."):
            loader = PDFPlumberLoader(f"uploads/{filename}")
            docs = loader.load()
            text_splitter = SemanticChunker(HuggingFaceEmbeddings())
            documents = text_splitter.split_documents(docs)

            if "vector_store" not in st.session_state or st.session_state.vector_store is None:
                embedder = HuggingFaceEmbeddings()
                st.session_state.vector_store = FAISS.from_documents(documents, embedder)
            else:
                st.session_state.vector_store.add_documents(documents)

            st.session_state.vector_store.save_local("vector_store")
            st.success(f"Processed and added {filename} to the vector store!")
    except Exception as e:
        st.error(f"Error processing {filename}: {e}")

# Owner Portal Logic
if mode == "Owner Portal":
    st.title("📂 Owner's Portal: PDF Management")

    with st.container():
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader("Drag and drop a PDF file here", type="pdf", key="uploader")

        if uploaded_file is not None:
            uploaded_filename = uploaded_file.name
            unique_id = uuid.uuid4().hex
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"{uploaded_filename}_{timestamp}_{unique_id}.pdf"

            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            with open(f"uploads/{saved_filename}", "wb") as f:
                f.write(uploaded_file.getvalue())

            process_and_add_to_vector_store(saved_filename)
            st.success(f"Uploaded PDF: **{uploaded_filename}**")

    st.header("📄 Uploaded PDFs")
    uploaded_pdfs = list_uploaded_pdfs()
    if uploaded_pdfs:
        for filename in uploaded_pdfs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {filename}")
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{filename}"):
                    delete_pdf(filename)
                    st.rerun()
    else:
        st.info("No PDFs uploaded yet.")

    st.header("🔍 Vector Store Status")
    if os.path.exists("vector_store"):
        st.success("✅ Vector store is up-to-date.")
    else:
        st.error("❌ Vector store not found. Please upload and process a PDF.")

# Chatbot Logic
elif mode == "Chatbot":
    st.title("Chatbot Assistant")
    st.markdown("Your AI partner, ready to help you")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    if st.session_state.vector_store is None:
        st.error("Vector store not found. Please upload PDFs in the Owner Portal.")
        st.stop()

    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "score_threshold": 0.7}
    )

    # Use Hugging Face model instead of Ollama
    hf_pipeline = pipeline(
        "text-generation",
        model="distilgpt2",  # Lightweight model for demo; replace with a better one if needed
        max_new_tokens=50,   # Limit response length
        truncation=True
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = """
    You are an AI assistant trained to answer questions based on the provided context.
    Follow these rules strictly:
    1. Use the context to answer the question.
    2. If the context is insufficient, say "I don't know."
    3. Provide only one sentence as the answer. Do not exceed one sentence under any circumstances.
    4. Do not include any reasoning, thoughts, or <think> tags in the answer.
    5. If the question is unrelated to the context, say "I don't know."
    Context: {context}
    Question: {question}
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True,
    )

    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🔍 Searching for answers..."):
            result = qa(user_input)
            response = result.get("result", "I don't know.")
            if not result.get("source_documents"):
                response = "I don't know."
            else:
                response = response.split(".")[0] + "."

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            if result.get("source_documents"):
                with st.expander("📚 View Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(f"📄 Source: {doc.metadata['source']}")
                        st.write("---")
