"""
rag_pipeline.py - RAG retrieval and LLM generation pipeline

This module orchestrates:
- Query embedding generation
- Similarity search in vector store
- Context preparation for LLM
- Answer generation using OpenAI GPT
- Source citation formatting
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from openai import OpenAI
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from utils import logger, format_source_citation


class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) pipeline
    Combines retrieval and generation for question answering
    """
    
    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            api_key: OpenAI API key
            embedding_model: Model for generating embeddings
            llm_model: Model for text generation
            temperature: Temperature for LLM generation (0-1)
            top_k: Number of documents to retrieve
        """
        self.api_key = api_key
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize OpenAI client for LLM calls
        self.llm_client = OpenAI(api_key=api_key)
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(api_key, embedding_model)
        
        # Vector store will be set later
        self.vector_store_manager = None
        
        logger.info(f"Initialized RAG pipeline with LLM: {llm_model}")
    
    def set_vector_store(self, vector_store_manager: VectorStoreManager):
        """
        Set the vector store manager
        
        Args:
            vector_store_manager: Initialized vector store manager
        """
        self.vector_store_manager = vector_store_manager
        logger.info("Vector store connected to RAG pipeline")
    
    def retrieve_context(self, query: str) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Retrieve relevant context chunks for a query
        
        Args:
            query: User query
        
        Returns:
            List of (distance, chunk_data) tuples
        """
        if self.vector_store_manager is None:
            raise ValueError("Vector store not set. Call set_vector_store() first.")
        
        # Generate query embedding
        logger.info(f"Generating embedding for query: {query}")
        query_embedding = self.embedding_manager.get_query_embedding(query)
        
        # Search vector store
        logger.info(f"Searching for top {self.top_k} relevant chunks")
        results = self.vector_store_manager.search(query_embedding, top_k=self.top_k)
        
        return results
    
    def _create_qa_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for question answering
        
        Args:
            query: User question
            context_chunks: List of relevant context chunks
        
        Returns:
            Formatted prompt string
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', 'N/A')
            text = chunk['text']
            
            context_parts.append(
                f"[Source {i+1}: {source}, Page {page}]\n{text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and accurate
- Cite which source(s) you used to answer (e.g., "According to Source 1...")

Answer:"""
        
        return prompt
    
    def _create_summary_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for document summarization
        
        Args:
            context_chunks: List of document chunks
        
        Returns:
            Formatted prompt string
        """
        # Combine all text chunks
        full_text = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        prompt = f"""You are a helpful assistant that creates concise summaries of documents.

Document Text:
{full_text}

Instructions:
- Create a clear and concise summary of the main points
- Organize the summary into key themes or sections
- Highlight the most important information
- Keep the summary informative but brief (3-5 paragraphs)

Summary:"""
        
        return prompt
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        Args:
            query: User question
        
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Processing question: {query}")
        
        # Step 1: Retrieve relevant context
        results = self.retrieve_context(query)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer this question.",
                'sources': [],
                'confidence': 'low'
            }
        
        # Extract chunks from results
        context_chunks = [result[1] for result in results]
        
        # Step 2: Create prompt
        prompt = self._create_qa_prompt(query, context_chunks)
        
        # Step 3: Generate answer using LLM
        logger.info("Generating answer with LLM")
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Step 4: Prepare sources
            sources = []
            for i, (distance, chunk_data) in enumerate(results):
                sources.append({
                    'source_id': i + 1,
                    'source_name': chunk_data['metadata'].get('source', 'Unknown'),
                    'page': chunk_data['metadata'].get('page', 'N/A'),
                    'text': chunk_data['text'][:200] + "...",  # Preview
                    'distance': float(distance)
                })
            
            logger.info("Successfully generated answer")
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': 'high' if len(results) >= 3 else 'medium'
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'confidence': 'error'
            }
    
    def summarize_document(self, max_chunks: int = 20) -> Dict[str, Any]:
        """
        Generate a summary of the entire document
        
        Args:
            max_chunks: Maximum number of chunks to include in summary
        
        Returns:
            Dict with summary and metadata
        """
        logger.info("Generating document summary")
        
        if self.vector_store_manager is None:
            raise ValueError("Vector store not set")
        
        # Get all chunks (or sample if too many)
        # We'll use a dummy query to get representative chunks
        dummy_query = "main points key information summary"
        query_embedding = self.embedding_manager.get_query_embedding(dummy_query)
        
        # Retrieve chunks
        results = self.vector_store_manager.search(query_embedding, top_k=max_chunks)
        
        if not results:
            return {
                'summary': "No document content available for summarization.",
                'num_chunks_used': 0
            }
        
        # Extract chunks
        context_chunks = [result[1] for result in results]
        
        # Create prompt
        prompt = self._create_summary_prompt(context_chunks)
        
        # Generate summary
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear and concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=800
            )
            
            summary = response.choices[0].message.content.strip()
            
            logger.info("Successfully generated summary")
            
            return {
                'summary': summary,
                'num_chunks_used': len(context_chunks),
                'sources': list(set([chunk['metadata'].get('source', 'Unknown') for chunk in context_chunks]))
            }
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'num_chunks_used': 0,
                'sources': []
            }
    
    def custom_query(self, query: str, system_prompt: str = None) -> str:
        """
        Execute a custom query with optional system prompt
        
        Args:
            query: User query
            system_prompt: Optional custom system prompt
        
        Returns:
            LLM response
        """
        # Retrieve context
        results = self.retrieve_context(query)
        
        if not results:
            return "No relevant context found for this query."
        
        # Build context
        context_chunks = [result[1] for result in results]
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers questions based on provided context."
        
        # Create user message with context
        user_message = f"Context:\n{context}\n\nQuery: {query}"
        
        # Generate response
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error in custom query: {e}")
            return f"Error: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        # Initialize pipeline
        pipeline = RAGPipeline(api_key)
        
        print("RAG Pipeline initialized successfully")
        print(f"LLM Model: {pipeline.llm_model}")
        print(f"Temperature: {pipeline.temperature}")
        print(f"Top-K retrieval: {pipeline.top_k}")
    else:
        print("Please set OPENAI_API_KEY in your environment")
