# requirements.txt
langchain==0.1.0
chromadb==0.4.18
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
openai==1.3.0
tiktoken==0.5.1
python-multipart==0.0.6
pydantic==2.4.2

# main.py
from fastapi import FastAPI, UploadFile, File
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize components
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
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
