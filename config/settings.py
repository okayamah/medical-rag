"""
Medical RAG System Configuration Settings
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """System configuration settings"""
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b-instruct-q4_0"
    # OLLAMA_MODEL: str = "llama3.2:3b-instruct-q4_0"
    
    # Vector Database Settings
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "medical_docs"
    
    # Embedding Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    
    # PubMed API Settings
    PUBMED_EMAIL: Optional[str] = "****"
    PUBMED_TOOL_NAME: str = "MedicalRAG"
    PUBMED_MAX_RESULTS: int = 500
    
    # # Langfuse Monitoring
    # LANGFUSE_SECRET_KEY: Optional[str] = None
    # LANGFUSE_PUBLIC_KEY: Optional[str] = None
    # LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Performance Settings
    SEARCH_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    GENERATION_TIMEOUT: int = 120


# Global settings instance
settings = Settings()