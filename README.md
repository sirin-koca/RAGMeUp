# RAGMeUp

**RAG** - an advanced AI-powered chatbot solution leveraging the Retrieval-Augmented Generation (RAG) methodology. 
This involves integrating state-of-the-art artificial intelligence techniques to enhance user interaction and deliver accurate responses in real-time.

**RAGMeUp** is a demo project for Retrieval-Augmented Generation (RAG) chatbot designed to answer questions specifically about NEET (Not in Employment, Education, or Training) based on a dataset of 130 PDFs. It retrieves relevant information from the PDFs and generates accurate, domain-specific responses. 

## Project Overview

The goal of RAGMeUp is to:
- Handle health-related queries focusing on the NEET domain.
- Provide accurate, verifiable answers based on the provided dataset (PDF files).
- Minimize hallucinations by limiting answers to only content found in the NEET dataset.
- Support both English and Norwegian language queries, responding in the language of the question.

This project uses **Streamlit** for the frontend, and **LangChain**, **Milvus**, and **Huggingface models** for the backend, including embedding models to vectorize the PDF content.

## Features

- Multilingual Support (English and Norwegian)
- Retrieval-Augmented Generation (RAG) for accurate query responses
- Simple and intuitive UI built with Streamlit
- Tested with both RAG and non-RAG approaches for performance comparison
- Low hallucination rate by limiting responses to NEET-related data

## Project Structure

```


RAGMeUp/
│
├── backend/
│   ├── backend.py          # Main logic: load PDFs, process embeddings, interact with Milvus
│   ├── popDB.py            # Populates the vector database (Milvus) with embeddings
│   ├── setupLLM.py         # Initializes and configures the LLM
│   └── milvus_utils.py     # Utility file to interact with Milvus (e.g., store, search vectors)
│
├── frontend/
│   └── app.py              # Web interface, e.g., Streamlit or Flask for web-based access
│
├── data/
│   ├── neet/               # Folder to store the NEET PDFs
│   └── embeddings/         # Folder to store the temporary embedding files 
│
├── utils/
│   └── helpers.py          # Utility functions, e.g., PDF loading, text cleaning, or tokenization
│
├── /images                   # Static images if needed
│   └── logo.png              # Optional: Navatar-Helper logo
│
├── .gitignore                # Exclude unnecessary files like __pycache__/, .env, etc.
├── README.md                 # Project documentation and setup instructions
├── requirements.txt          # Dependencies (e.g., LangChain, Milvus, Streamlit)

```

### Prerequisites
- Python 3.8 or higher
- PyCharm or any Python IDE
- A virtual environment for dependency management
- Git for version control




