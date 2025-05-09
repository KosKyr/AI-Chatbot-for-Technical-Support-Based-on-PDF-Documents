# app.py

import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# === Load Vectorstore ===
@st.cache_resource
def load_vectorstore(chroma_path="./chroma_db"):
    embedding = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    return Chroma(persist_directory=chroma_path, embedding_function=embedding)

# === Load FLAN-T5 Model ===
@st.cache_resource
def load_qa_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        device=-1  # Use CPU
    )

# === Generate Answer ===
def generate_response(question, retriever, qa_model):
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find relevant information.", []

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Use the following context to answer the question. Base your answer only on the provided information.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_model(prompt, max_length=512, do_sample=False)[0]["generated_text"].strip()
    return result, docs

# === Streamlit App ===
def main():
    st.set_page_config(page_title="📄 PDF QA Chatbot", layout="wide")
    st.title("📄 PDF Chatbot (Context-Aware)")

    with st.sidebar:
        st.markdown("### 🔧 Settings")
        chroma_path = st.text_input("Chroma DB Path", "./chroma_db")

    # Load resources
    with st.spinner("Loading vectorstore..."):
        vectorstore = load_vectorstore(chroma_path)
    with st.spinner("Loading model..."):
        qa_model = load_qa_model()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = st.text_input("Ask a question about your PDFs:")
    if question:
        with st.spinner("Generating answer..."):
            answer, sources = generate_response(question, retriever, qa_model)
            st.markdown("### 💬 Answer")
            st.write(answer)

            st.markdown("### 📚 Sources")
            for i, doc in enumerate(sources, 1):
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                st.markdown(f"**{i}.** Source: `{src}` (page {page})")
                st.markdown(f"> {doc.page_content[:500]}...")

if __name__ == "__main__":
    main()