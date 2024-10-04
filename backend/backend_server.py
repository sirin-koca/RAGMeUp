# backend_server.py
# We're using faiss-cpu for local testing. When deploying to the GPU server, we'll switch to faiss-gpu.

from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# Define the request model
class QueryRequest(BaseModel):
    question: str


# Initialize the FastAPI app
app = FastAPI()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16
)

# Set up the text generation pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_length=512,
    do_sample=True,
    top_p=0.95,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=generator)

# Prepare the knowledge base (we'll create sample documents in Step 1.4)
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

loader = DirectoryLoader('../data/documents', glob='*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Initialize embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(docs, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)


# Define the API endpoint
@app.post("/get_answer")
async def get_answer(request: QueryRequest):
    question = request.question
    answer = qa_chain.run(question)
    return {"answer": answer}
