from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    app_name: str = "RAG PDF System"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    
    # Vector Store (FAISS)
    vector_db_path: str = "./faiss_db"
    collection_name: str = "pdf_documents"
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()