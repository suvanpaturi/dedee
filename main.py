from fastapi import FastAPI, UploadFile, File
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize components
model_name = os.environ.get("TINYLLAMA_PATH", "/app/models/tinyllama")
embedding_model_name = os.environ.get("ST_PATH", "/app/models/sentence-transformer")

# Load tokenizer and model - CPU optimized configuration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Create language model pipeline optimized for CPU
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,  # Reduced for CPU memory constraints
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15,
    do_sample=True
)

# Initialize Langchain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Initialize embeddings with CPU configuration
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 8}  # Smaller batch size for CPU
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Smaller chunks for CPU processing
    chunk_overlap=150,
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
    
    return {"message": "Document processed successfully", "chunks": len(texts)}

@app.post("/query")
async def query(question: str):
    if not vector_store:
        return {"error": "No documents have been uploaded yet"}
    
    # Create QA chain with CPU-optimized settings
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),  # Reduced k for CPU
    )
    
    # Get response
    response = qa_chain.run(question)
    
    return {"answer": response}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify models are loaded correctly"""
    model_status = "loaded" if model is not None else "not loaded"
    embedding_status = "initialized" if embeddings is not None else "not initialized"
    return {
        "status": "healthy",
        "model": model_status,
        "embeddings": embedding_status,
        "model_path": model_name,
        "embedding_path": embedding_model_name
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)