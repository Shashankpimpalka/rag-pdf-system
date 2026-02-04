from pydantic import BaseModel, Field
from typing import List, Optional

class PDFUploadResponse(BaseModel):
    filename: str
    num_pages: int
    num_chunks: int
    status: str
    message: str

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the PDFs")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve")

class Source(BaseModel):
    content: str
    page: int
    filename: str
    score: float

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    context_used: bool

class HealthResponse(BaseModel):
    status: str
    message: str
    ollama_status: str
    vector_db_status: str