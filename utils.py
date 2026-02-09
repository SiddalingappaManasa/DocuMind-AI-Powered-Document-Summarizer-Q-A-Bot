"""
utils.py - Helper functions and utilities

This module contains:
- Text cleaning and preprocessing
- Token counting
- Configuration management
- Logging setup
"""

import os
import re
import logging
from typing import List
import tiktoken
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def setup_logging():
    """
    Configure logging for the application
    Returns a logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def get_config():
    """
    Load configuration from environment variables
    Returns a dict with all configuration settings
    """
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4-turbo-preview'),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'vector_db_path': os.getenv('VECTOR_DB_PATH', './vector_db'),
        'chunk_size': int(os.getenv('CHUNK_SIZE', '800')),
        'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
        'top_k_results': int(os.getenv('TOP_K_RESULTS', '5'))
    }
    
    # Validate API key
    if not config['openai_api_key']:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return config


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (default: gpt-4)
    
    Returns:
        Number of tokens
    """
    try:
        # Get the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model)
        
        # Encode the text and count tokens
        tokens = encoding.encode(text)
        return len(tokens)
    
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence splitter using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def ensure_directory_exists(directory: str):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def format_source_citation(chunk_text: str, metadata: dict) -> str:
    """
    Format a source citation for display
    
    Args:
        chunk_text: Text chunk content
        metadata: Metadata dict with source info
    
    Returns:
        Formatted citation string
    """
    source = metadata.get('source', 'Unknown')
    page = metadata.get('page', 'N/A')
    
    # Get a preview of the chunk (first 100 chars)
    preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
    
    citation = f"**Source:** {source}"
    if page != 'N/A':
        citation += f" (Page {page})"
    citation += f"\n**Excerpt:** {preview}\n"
    
    return citation


def validate_file_type(filename: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate file type by extension
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (default: pdf, docx, txt)
    
    Returns:
        True if file type is valid, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.docx', '.txt']
    
    # Get file extension
    _, ext = os.path.splitext(filename.lower())
    
    return ext in allowed_extensions


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


# Example usage and testing
if __name__ == "__main__":
    # Test token counting
    sample_text = "This is a sample text for testing token counting."
    token_count = count_tokens(sample_text)
    print(f"Text: {sample_text}")
    print(f"Token count: {token_count}")
    
    # Test text cleaning
    dirty_text = "This   has    extra   spaces!!!   @#$"
    clean = clean_text(dirty_text)
    print(f"\nOriginal: {dirty_text}")
    print(f"Cleaned: {clean}")
    
    # Test configuration loading
    try:
        config = get_config()
        print(f"\nConfiguration loaded successfully")
        print(f"Embedding model: {config['embedding_model']}")
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
