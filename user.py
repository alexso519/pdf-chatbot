import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import os

# Custom CSS for modern UI
st.markdown("""
    <style>
        body {
            font-family: 'Noto Sans TC', sans-serif;
            background-color: #f0f4f8;
        }
        .chat-container {
            height: calc(100vh - 50px);
            max-height: 80vh;
            background-color: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
            margin: 0 auto;
            width: 100%; 
            max-width: 850px; 
        }
        .chat-content {
            height: 100%;
            overflow-y: auto;
            padding: 10px;
        }
        header {
            text-align: center;
            margin-bottom: 1rem;
        }
        h1 {
            font-size: 2rem;
            font-weight: bold;
            color: #4a4a4a;
            margin-bottom: 0.5rem;
        }
        p {
            font-size: 1rem;
            color: #6b7280;
        }
        @media (max-width: 640px) {
            .chat-container {
                height: calc(100vh - 100px);
            }
            h1 {
                font-size: 1.8rem;
            }
            p {
                font-size: 0.9rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("Chatbot Assistant")
st.markdown("Your AI partner, ready to help you")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input at the beginning
user_input = st.chat_input("Type your question here...")

# Load vector store without displaying information
if os.path.exists("vector_store"):
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.load_local(
        "vector_store",
        embedder,
        allow_dangerous_deserialization=True  # Allow deserialization
    )
else:
    st.stop()  # Stop the app if vector store is not found

# Define retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "score_threshold": 0.7}  # Adjust score threshold
)

# Define LLM and prompt
llm = Ollama(model="deepseek-r1:1.5b")
prompt = """
You are an AI assistant trained to answer questions based on the provided context.
Follow these rules strictly:
1. Use the context to answer the question.
2. If the context is insufficient, say "I don't know."
3. Do not include any reasoning, thoughts, or <think> tags in the answer.
4. If the question is unrelated to the context, say "I don't know."

Context: {context}
Question: {question}
Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# Define chains
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

# Process input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🔍 Searching for answers..."):
        result = qa(user_input)
        response = result.get("result", "I don't know.")  # Default response

        # Check if source documents are empty
        if not result.get("source_documents"):
            response = "I don't know."
        else:
            # Enforce one-sentence response
            if "<think>" in response:
                response = response.split("</think>")[-1].strip()
            response = response.split(".")[0] + "."

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Source documents expander
        if result.get("source_documents"):
            with st.expander("📚 View Source Documents"):
                source_documents = result["source_documents"]
                for doc in source_documents:
                    st.write(f"📄 Source: {doc.metadata['source']}")
                    st.write("---")