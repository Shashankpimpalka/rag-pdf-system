from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path

from app.config import get_settings
from app.models.schemas import (
    PDFUploadResponse, 
    QuestionRequest, 
    QuestionResponse,
    HealthResponse
)
from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStoreService
from app.services.qa_service import QAService

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based PDF Question Answering System using LangChain and FAISS"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Angular app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStoreService()
qa_service = QAService(vector_store)

# ============= ENDPOINTS =============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 RAG PDF System API",
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": {
            "upload_pdf": "/upload-pdf",
            "ask_question": "/ask",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint - verifies Ollama and vector store status"""
    try:
        # Check Ollama status
        ollama_status = qa_service.check_ollama_status()
        
        # Check vector store
        stats = vector_store.get_collection_stats()
        
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            ollama_status=ollama_status.get('status', 'unknown'),
            vector_db_status="connected" if stats.get('total_documents', 0) >= 0 else "error"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/upload-pdf", response_model=PDFUploadResponse, tags=["PDF Processing"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file
    
    - **file**: PDF file to upload and process
    
    Returns information about the processed PDF including number of pages and chunks created.
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    file_path = None
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📥 Received file: {file.filename}")
        
        # Process PDF
        chunks, num_pages = pdf_processor.process_pdf(str(file_path))
        
        # Add to vector store
        num_chunks = vector_store.add_documents(chunks)
        
        print(f"✅ Successfully processed {file.filename}")
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return PDFUploadResponse(
            filename=file.filename,
            num_pages=num_pages,
            num_chunks=num_chunks,
            status="success",
            message=f"Successfully processed {num_pages} pages into {num_chunks} chunks"
        )
        
    except Exception as e:
        # Clean up on error
        if file_path and file_path.exists():
            os.remove(file_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.post("/ask", response_model=QuestionResponse, tags=["Question Answering"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded PDFs
    
    - **question**: Your question about the documents
    - **top_k**: Number of relevant chunks to retrieve (default: 3, max: 10)
    
    Returns an answer based on the content of uploaded PDFs with source citations.
    """
    try:
        print(f"\n❓ Question: {request.question}")
        
        response = qa_service.answer_question(
            question=request.question,
            top_k=request.top_k
        )
        
        print(f"✅ Answer generated")
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error answering question: {str(e)}"
        )

@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Get vector store statistics - shows how many documents are stored"""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "vector_store": stats,
            "model": settings.ollama_model,
            "embedding_model": settings.embedding_model
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )

@app.delete("/clear-database", tags=["Administration"])
async def clear_database():
    global vector_store, qa_service
    """⚠️ Clear all documents from vector store - THIS CANNOT BE UNDONE"""
    try:
        vector_store.delete_collection()
        
        # Reinitialize vector store
        
        vector_store = VectorStoreService()
        qa_service = QAService(vector_store)
        
        return {
            "status": "success",
            "message": "Vector store cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing database: {str(e)}"
        )

# ============= ERROR HANDLERS =============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

# ============= STARTUP/SHUTDOWN =============

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║  🚀 RAG PDF System Started                          ║
    ║  📚 Vector Store: {settings.collection_name:30s} ║
    ║  🤖 LLM Model: {settings.ollama_model:33s} ║
    ║  🧠 Embeddings: {settings.embedding_model:31s} ║
    ║  🌐 Server: http://{settings.host}:{settings.port:<27d} ║
    ╚══════════════════════════════════════════════════════╝
    """)

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("\n👋 Shutting down RAG PDF System...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )