import os
from huggingface_hub import HfFolder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# Set Hugging Face token and cache directory
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
os.environ["TRANSFORMERS_CACHE"] = "/groups/rag2/RAGMeUp/hf_cache"

# Override Hugging Face cache directory inside the script
HfFolder.path = lambda: "/groups/rag2/RAGMeUp/hf_cache"


# Function to load, split, and embed PDFs
def populate_milvus():
    # Load PDFs
    pdf_paths = [
        "./data/neet/NEET1YoungInTodaysLabourMarketOffer.pdf",
        "./data/neet/NEET2EthnicityGenderHouseholdNEETIntersectional.pdf",
        "./data/neet/NEET3YouthMentalHealthServices.pdf"
    ]

    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = splitter.split_documents(documents)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Store vectors in Milvus
    vector_db = Milvus.from_documents(docs, embeddings, collection_name="neet_collection")

    print("Milvus database populated with NEET embeddings")


if __name__ == "__main__":
    populate_milvus()
