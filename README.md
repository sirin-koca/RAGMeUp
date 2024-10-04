# RAGMeUp

RAGMeUp is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions specifically about NEET (Not in Employment, Education, or Training) based on a dataset of 130 PDFs. It retrieves relevant information from the PDFs and generates accurate, domain-specific responses. 

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

```bash
RAGMeUp/
│
├── backend/               # Backend logic for processing queries
│   └── backend.py
│
├── frontend/              # Streamlit frontend for user interaction
│   └── client.py
│
├── util/                  # Helper functions (e.g., text extraction, language detection)
│   └── helpers.py
│
├── data/                  # Directory to store PDF files
│   └── sample_pdf1.pdf
│   └── sample_pdf2.pdf
│   └── sample_pdf3.pdf
│
├── images/                # (Optional) Static assets like logo or images for the frontend
│
├── app.py                 # Main entry point for the Streamlit app
├── .gitignore             # Ignored files/folders (e.g., __pycache__/)
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies
