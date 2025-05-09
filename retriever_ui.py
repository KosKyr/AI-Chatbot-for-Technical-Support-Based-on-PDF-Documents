import streamlit as st
from retriever import DocumentLoader, TextSplitter, ChromaDB, DBBuilder, QueryEngine

st.set_page_config(page_title="PDF Search", layout="wide")
st.title("📄 PDF Document Search Engine")

# Load and build the vector store
loader = DocumentLoader(data_path="./knowledge")
splitter = TextSplitter(chunk_size=900, chunk_overlap=400)
chroma_db = ChromaDB(db_path="./chroma_db", model_name="multi-qa-MiniLM-L6-cos-v1")
db_builder = DBBuilder(loader, splitter, chroma_db)
vectorstore = db_builder.build_or_load()
engine = QueryEngine(vectorstore)

# Sidebar toggle
mode = st.sidebar.radio("Choose Mode", ["🔍 Search", "📂 View Database"])

if mode == "🔍 Search":
    with st.form("query_form"):
        user_query = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Search")

    if submitted and user_query.strip():
        results = engine.search(user_query)

        if results:
            st.markdown(f"### 🔎 Search Results for: *{user_query}*")
            st.write(f"Found {len(results)} relevant chunks:")
            for i, doc in enumerate(results, 1):
                st.markdown(f"#### Result {i}")
                st.write(doc.page_content)
                st.markdown("---")
        else:
            st.warning("No results found.")

elif mode == "📂 View Database":
    st.markdown("### 🗃️ All Document Chunks in Vectorstore")

    try:
        # Display all chunks in the database
        docs = vectorstore.get()["documents"]  # Direct access to raw chunks
        metadatas = vectorstore.get()["metadatas"]
        ids = vectorstore.get()["ids"]

        for i, (doc, meta, doc_id) in enumerate(zip(docs, metadatas, ids), 1):
            st.markdown(f"#### Chunk {i} (ID: `{doc_id}`)")
            st.write(doc)
            if meta:
                st.caption(f"Metadata: {meta}")
            st.markdown("---")
    except Exception as e:
        st.error(f"Failed to fetch documents: {e}")


# import streamlit as st
# from retriever import DocumentLoader, TextSplitter, ChromaDB, DBBuilder, QueryEngine

# st.set_page_config(page_title="PDF Search", layout="wide")
# st.title(" PDF Document Search Engine")

# loader = DocumentLoader(data_path="./knowledge")
# splitter = TextSplitter(chunk_size=1000, chunk_overlap=400)
# chroma_db = ChromaDB(db_path="./chroma_db", model_name="all-MiniLM-L6-v2")
# db_builder = DBBuilder(loader, splitter, chroma_db)
# vectorstore = db_builder.build_or_load()
# engine = QueryEngine(vectorstore)

# with st.form("query_form"):
#     user_query = st.text_input("Enter your question:")
#     submitted = st.form_submit_button("Search")

# if submitted and user_query.strip():
#     results = engine.search(user_query)
#     if results:
#         for i, doc in enumerate(results, 1):
#             st.markdown(f"### Result {i}")
#             st.write(doc.page_content)
#             st.markdown("---")
#     else:
        st.warning("No results found.")