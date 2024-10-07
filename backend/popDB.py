from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

# Define paths
pdf_dir = "data/neet/"  # Directory where your NEET PDFs are stored
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"  # Example embedding model
milvus_collection_name = "NEET_Embeddings"

# Initialize text splitter and embedding model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)


# Load PDFs and process
def process_pdfs():
    pdf_files = ["NEET1.pdf", "NEET2.pdf"]  # Add your actual PDF file names here
    documents = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(f"{pdf_dir}/{pdf_file}")
        # Load and split the PDF content
        doc = loader.load()
        chunks = text_splitter.split_documents(doc)
        documents.extend(chunks)

    # Generate embeddings for the documents
    embeddings = embedder.embed_documents([doc.page_content for doc in documents])

    # Store embeddings in Milvus
    milvus = Milvus.from_documents(documents, embedding=embedder, collection_name=milvus_collection_name)

    print("PDFs processed and embeddings stored successfully in Milvus.")


if __name__ == "__main__":
    process_pdfs()
