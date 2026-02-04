from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List
import os
from app.config import get_settings

settings = get_settings()

class PDFProcessor:
    """Process PDF files and split into chunks"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF and extract text"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Add filename to metadata
            filename = os.path.basename(file_path)
            for page in pages:
                page.metadata['filename'] = filename
                page.metadata['source'] = file_path
            
            print(f"📄 Loaded {len(pages)} pages from {filename}")
            return pages
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk information to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
            
            print(f"✂️  Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")
    
    def process_pdf(self, file_path: str) -> tuple[List[Document], int]:
        """
        Complete PDF processing pipeline
        Returns: (chunks, num_pages)
        """
        # Load PDF
        pages = self.load_pdf(file_path)
        num_pages = len(pages)
        
        # Split into chunks
        chunks = self.split_documents(pages)
        
        return chunks, num_pages
    
    def extract_metadata(self, chunks: List[Document]) -> dict:
        """Extract useful metadata from chunks"""
        if not chunks:
            return {}
        
        return {
            "total_chunks": len(chunks),
            "filename": chunks[0].metadata.get('filename', 'unknown'),
            "pages": list(set(chunk.metadata.get('page', 0) for chunk in chunks))
        }