from dataclasses import dataclass 
from openai import AsyncOpenAI
from RAG import RAGPipeline

@dataclass 
class Dependencies:
    
    api_client: AsyncOpenAI
    rag_pipeline: RAGPipeline