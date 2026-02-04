from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Tuple
from app.config import get_settings
import os
import pickle

settings = get_settings()


class VectorStoreService:
    """Manage vector database operations using FAISS"""
    
    
    def __init__(self):
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create directory if doesn't exist
        os.makedirs(settings.vector_db_path, exist_ok=True)
        
        # Paths for FAISS index and metadata
        self.index_path = os.path.join(settings.vector_db_path, "faiss_index")
        self.metadata_path = os.path.join(settings.vector_db_path, "metadata.pkl")
        
        # Initialize or load vector store
        self.vector_store = self._initialize_vector_store()
        self.document_count = 0
    
    def _initialize_vector_store(self) -> FAISS:
        """Initialize or load existing FAISS vector store"""
        try:
            # Try to load existing store
            if os.path.exists(self.index_path) and os.path.exists(f"{self.index_path}.faiss"):
                vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load document count
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        self.document_count = metadata.get('count', 0)
                
                print(f"✅ Loaded existing FAISS index from {self.index_path}")
                print(f"   Documents in store: {self.document_count}")
                return vector_store
        except Exception as e:
            print(f"⚠️  Could not load existing index: {str(e)}")
        
        # Create new store with a dummy document
        print(f"⚠️  Creating new FAISS index")
        dummy_doc = Document(
            page_content="Initialization document",
            metadata={"type": "init"}
        )
        vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
        self.document_count = 0
        return vector_store
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to vector store
        Returns: number of documents added
        """
        try:
            if not documents:
                return 0
            
            # Add documents to existing store
            self.vector_store.add_documents(documents)
            
            # Update count
            self.document_count += len(documents)
            
            # Save to disk
            self._persist()
            
            return len(documents)
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    def _persist(self):
        """Persist FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            self.vector_store.save_local(self.index_path)
            
            # Save metadata
            metadata = {
                'count': self.document_count,
                'embedding_model': settings.embedding_model
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"💾 Persisted {self.document_count} documents to {self.index_path}")
        except Exception as e:
            print(f"❌ Error persisting vector store: {str(e)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents
        Returns: List of (document, similarity_score) tuples
        """
        try:
            if self.document_count == 0:
                return []
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=min(k, self.document_count)
            )
            
            # Filter out initialization document
            filtered_results = [
                (doc, score) for doc, score in results
                if doc.metadata.get('type') != 'init'
            ]
            
            return filtered_results
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store"""
        try:
            return {
                "total_documents": self.document_count,
                "collection_name": "faiss_index",
                "embedding_model": settings.embedding_model,
                "index_path": self.index_path
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            # Remove FAISS index files
            if os.path.exists(f"{self.index_path}.faiss"):
                os.remove(f"{self.index_path}.faiss")
            if os.path.exists(f"{self.index_path}.pkl"):
                os.remove(f"{self.index_path}.pkl")
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            # Reinitialize
            self.vector_store = self._initialize_vector_store()
            self.document_count = 0
            
            print(f"🗑️  Deleted FAISS collection")
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")
    
    def search_by_filename(self, filename: str, k: int = 5) -> List[Document]:
        """Get documents from a specific file"""
        try:
            if self.document_count == 0:
                return []
            
            # Get all documents and filter by filename
            all_docs = self.vector_store.similarity_search("", k=min(k*10, self.document_count))
            filtered = [
                doc for doc in all_docs 
                if doc.metadata.get('filename') == filename
                and doc.metadata.get('type') != 'init'
            ]
            return filtered[:k]
        except Exception as e:
            raise Exception(f"Error searching by filename: {str(e)}")