# requirements.txt
langchain==0.1.0
chromadb==0.4.18
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
tiktoken==0.5.1
python-multipart==0.0.6
pydantic==2.4.2
transformers==4.36.0
torch==2.1.0
accelerate==0.25.0
bitsandbytes==0.41.0
sentenceformers==2.2.2

# main.py
from fastapi import FastAPI, UploadFile, File
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize components
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)

# Create language model pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

# Initialize Langchain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Initialize vector store
vector_store = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    
    # Save uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Load and process document
    loader = TextLoader(file.filename)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    
    # Create or update vector store
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Clean up temporary file
    os.remove(file.filename)
    
    return {"message": "Document processed successfully"}

@app.post("/query")
async def query(question: str):
    if not vector_store:
        return {"error": "No documents have been uploaded yet"}
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    )
    
    # Get response
    response = qa_chain.run(question)
    
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
