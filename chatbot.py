# chatbot.py

import os
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# === Load Vectorstore ===
def load_vectorstore(chroma_path="./chroma_db"):
    embedding = SentenceTransformerEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embedding)
    print(f"✅ Loaded vectorstore from: {chroma_path}")
    return vectorstore

# === Chatbot Class ===
class Chatbot:
    def __init__(self, vectorstore):
        self.retriever: BaseRetriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        print("⏳ Loading FLAN-T5 model...")
        self.qa_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            tokenizer="google/flan-t5-base",
            device=-1  # Use CPU
        )
        print("✅ Model loaded.")

    def generate_response(self, question: str):
        # Retrieve top-k relevant documents
        docs = self.retriever.invoke(question)
        if not docs:
            return "I couldn't find relevant information.", []

        # Combine retrieved document content
        combined_context = "\n\n".join([doc.page_content for doc in docs])

        # More flexible prompt
        prompt = f"""
You are a helpful assistant. Use the context below to answer the question as clearly and informatively as possible.
If the context gives hints, try to form a thoughtful answer, even if it's not explicitly stated word for word.

Context:
{combined_context}

Question:
{question}

Answer:
"""

        # Run generation
        output = self.qa_model(prompt, max_length=512, do_sample=False)[0]['generated_text'].strip()
        return output, docs

# === Display Results ===
def display_chat(question, answer, sources):
    print("\n" + "=" * 60)
    print(f"❓ Question: {question}\n")
    print(f"💬 Answer:\n{answer}\n")
    print("📚 Sources used:")
    for i, doc in enumerate(sources, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n{i}. Source: {src} (page {page})")
        print(f"> {doc.page_content[:300].strip()}...")

# === Main Loop ===
def main():
    vectorstore = load_vectorstore("./chroma_db")
    chatbot = Chatbot(vectorstore)

    print("\n🧠 Chatbot Ready — type your question (or 'quit' to exit)\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        answer, sources = chatbot.generate_response(question)
        display_chat(question, answer, sources)

if __name__ == "__main__":
    main()
