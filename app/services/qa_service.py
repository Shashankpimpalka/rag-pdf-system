from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Tuple
from app.config import get_settings
from app.services.vector_store import VectorStoreService
from app.models.schemas import QuestionResponse, Source

settings = get_settings()

class QAService:
    """Question Answering service using RAG"""
    
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1
        )
        
        # Custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create custom prompt template for QA"""
        template = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.

Use the following pieces of context to answer the question at the end. 

Important instructions:
- If you can answer the question using the context, provide a detailed and accurate answer.
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
- Always cite which parts of the context you used.
- Be concise but comprehensive.
- If you mention specific details, try to include the page number if available.

Context:
{context}

Question: {question}

Detailed Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_rag_chain(self):
        """Create the RAG chain using LCEL (LangChain Expression Language)"""
        
        def format_docs(docs):
            """Format documents for context"""
            formatted = []
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get('page', 'unknown')
                filename = doc.metadata.get('filename', 'unknown')
                content = doc.page_content
                formatted.append(f"[Source {i} - {filename}, Page {page}]:\n{content}")
            return "\n\n".join(formatted)
        
        # Create retriever
        retriever = self.vector_store.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Build RAG chain using LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def answer_question(
        self, 
        question: str, 
        top_k: int = 3
    ) -> QuestionResponse:
        """
        Answer a question using RAG
        """
        try:
            # Get relevant documents first for sources
            similar_docs = self.vector_store.similarity_search(
                query=question,
                k=top_k
            )
            
            if not similar_docs:
                return QuestionResponse(
                    question=question,
                    answer="No relevant documents found in the database. Please upload PDFs first.",
                    sources=[],
                    context_used=False
                )
            
            print(f"🔍 Retrieved {len(similar_docs)} relevant chunks")
            print(f"💭 Generating answer...")
            
            # Generate answer using RAG chain
            answer = self.rag_chain.invoke(question)
            
            # Extract sources
            sources = self._extract_sources(similar_docs)
            
            return QuestionResponse(
                question=question,
                answer=answer,
                sources=sources,
                context_used=True
            )
            
        except Exception as e:
            raise Exception(f"Error answering question: {str(e)}")
    
    def _extract_sources(
        self, 
        docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Source]:
        """Extract source information"""
        sources = []
        
        for doc, score in docs_with_scores:
            # Truncate content for preview
            content = doc.page_content
            preview = content[:200] + "..." if len(content) > 200 else content
            
            sources.append(Source(
                content=preview,
                page=doc.metadata.get('page', 0),
                filename=doc.metadata.get('filename', 'unknown'),
                score=float(score)
            ))
        
        return sources
    
    def check_ollama_status(self) -> dict:
        """Check if Ollama is running and accessible"""
        try:
            # Try to generate a simple response
            response = self.llm.invoke("Say 'OK' if you can read this.")
            return {
                "status": "connected",
                "model": settings.ollama_model,
                "response": response[:50] if response else "No response"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }