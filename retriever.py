# --- Installation (uncomment to run in Colab) ---
# !pip install chromadb langchain langchain-community langchain-huggingface pypdf sentence-transformers
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated embedding import
from langchain_core.documents import Document  # Adjusted if needed by downstream code

class DocumentLoader:
    def __init__(self, data_path="./knowledge"):
        self.data_path = data_path

    def load(self):
        loader = DirectoryLoader(self.data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        return loader.load()

class TextSplitter:
    def __init__(self, chunk_size=2000, chunk_overlap=250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

class ChromaDB:
    def __init__(self, db_path="./chroma_db", model_name="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.model_name)
        self.vectorstore = None

    def from_documents(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.db_path
        )
        self.vectorstore.persist()
        return self.vectorstore

    def load(self):
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        return self.vectorstore

class DBBuilder:
    def __init__(self, loader: DocumentLoader, splitter: TextSplitter, db: ChromaDB):
        self.loader = loader
        self.splitter = splitter
        self.db = db

    def build_or_load(self):
        try:
            if not os.path.exists(os.path.join(self.db.db_path, "index")):
                documents = self.loader.load()
                chunks = self.splitter.split(documents)
                return self.db.from_documents(chunks)
            else:
                return self.db.load()
        except ValueError as e:
            if "tenant" in str(e):
                import shutil
                shutil.rmtree(self.db.db_path, ignore_errors=True)
                documents = self.loader.load()
                chunks = self.splitter.split(documents)
                return self.db.from_documents(chunks)
            else:
                raise e


class QueryEngine:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def search(self, query: str):
        return self.retriever.get_relevant_documents(query)

def printResults(results):
    answers = [doc.page_content for doc in results]
    print("\nTop 5 Relevant Chunks:\n" + "-"*30)
    for i, chunk in enumerate(answers, 1):
        print(f"\nChunk {i}:\n{chunk}\n{'-'*30}")

# --- Pipeline Setup ---
loader = DocumentLoader(data_path="./knowledge")
splitter = TextSplitter(chunk_size=900, chunk_overlap=400)
chroma_db = ChromaDB(db_path="./chroma_db", model_name="multi-qa-MiniLM-L6-cos-v1")
db_builder = DBBuilder(loader, splitter, chroma_db)
vectorstore = db_builder.build_or_load()
engine = QueryEngine(vectorstore)

# --- Questions to Search ---
questions = [
    "What features does MATLAB offer to help shorten response times and reduce data transmission over the network?",
    "How did Baker Hughes engineers use MATLAB to develop pump health monitoring software?",
    "Why is it important for training data in predictive maintenance systems to include instances from both normal and fault conditions?",
    "What is the recall performance of the proposed ENBANN method in comparison to other methods?",
    "What is cross-sectional prediction and how can it be applied in estimating component lifespan?",
    "Why are gas leak detectors important in environments with many pneumatic valves, and what type of detectors are considered non-intrusive?",
    "What new Industry 4.0 technologies are being used for remote asset monitoring, and what tools support them?",
    "What does the simulation model of the SUDM policy evaluate, and what assumptions are made about workstation operations?",
    "How were the prior parameters for the Weibull and exponential degradation models estimated, and what assumptions were made about the error terms?",
    "How does fuzzy logic contribute to diagnostics in machine failure and maintenance management?",
    "Why are artificial neural networks suitable for prognostics in machine failure, and what limitations do traditional systems face?",
    "How do Big Data platforms and CMMS contribute to the formulation of maintenance strategies?",
    "What is the relationship between diagnostics and prognostics in the context of machine degradation and failure?"
]

for question in questions:
    results = engine.search(question)
    printResults(results)
