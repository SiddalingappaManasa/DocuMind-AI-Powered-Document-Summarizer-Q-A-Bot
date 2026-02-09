"""
ingest.py - Document loading and text chunking

This module handles:
- Loading documents from PDF, DOCX, and TXT files
- Extracting text content
- Chunking text into smaller segments with overlap
- Preprocessing and cleaning text
"""

import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from docx import Document
from utils import clean_text, count_tokens, logger, ensure_directory_exists


class DocumentLoader:
    """
    Handles loading documents from various file formats
    """
    
    def __init__(self):
        """Initialize the document loader"""
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and extract text from a PDF file
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            List of dicts containing text and metadata for each page
        """
        documents = []
        
        try:
            # Open the PDF file
            pdf_reader = PdfReader(file_path)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Only add non-empty pages
                if text.strip():
                    documents.append({
                        'text': clean_text(text),
                        'metadata': {
                            'source': os.path.basename(file_path),
                            'page': page_num + 1,
                            'format': 'pdf'
                        }
                    })
            
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def load_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and extract text from a DOCX file
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            List with a single dict containing text and metadata
        """
        try:
            # Open the DOCX file
            doc = Document(file_path)
            
            # Extract text from all paragraphs
            full_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            
            documents = [{
                'text': clean_text(full_text),
                'metadata': {
                    'source': os.path.basename(file_path),
                    'page': 1,
                    'format': 'docx'
                }
            }]
            
            logger.info(f"Loaded DOCX file: {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise
    
    def load_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and extract text from a TXT file
        
        Args:
            file_path: Path to TXT file
        
        Returns:
            List with a single dict containing text and metadata
        """
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            documents = [{
                'text': clean_text(text),
                'metadata': {
                    'source': os.path.basename(file_path),
                    'page': 1,
                    'format': 'txt'
                }
            }]
            
            logger.info(f"Loaded TXT file: {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            raise
    
    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document based on its file extension
        
        Args:
            file_path: Path to the document
        
        Returns:
            List of dicts containing text and metadata
        """
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Route to appropriate loader
        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext == '.docx':
            return self.load_docx(file_path)
        elif ext == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class TextChunker:
    """
    Handles chunking text into smaller segments with overlap
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Target size for each chunk (in tokens)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
        
        Returns:
            List of dicts containing chunk text and metadata
        """
        chunks = []
        
        # Split text into words for easier chunking
        words = text.split()
        
        # Estimate tokens per word (rough approximation: 1 word ≈ 1.3 tokens)
        words_per_chunk = int(self.chunk_size / 1.3)
        words_overlap = int(self.chunk_overlap / 1.3)
        
        # Create chunks with overlap
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            # Get end index for this chunk
            end_idx = min(start_idx + words_per_chunk, len(words))
            
            # Extract chunk words and join into text
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Verify token count is within range
            token_count = count_tokens(chunk_text)
            
            # Create chunk dict
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': chunk_id,
                    'token_count': token_count
                }
            })
            
            chunk_id += 1
            
            # Move start index forward (with overlap)
            start_idx = end_idx - words_overlap
            
            # Break if we've processed all words
            if end_idx >= len(words):
                break
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dicts with text and metadata
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['text'], doc['metadata'])
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


class DocumentIngestor:
    """
    Main class that orchestrates document loading and chunking
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the document ingestor
        
        Args:
            chunk_size: Target size for each chunk (in tokens)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
    
    def ingest_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and chunk a document
        
        Args:
            file_path: Path to the document file
        
        Returns:
            List of chunks with text and metadata
        """
        logger.info(f"Ingesting document: {file_path}")
        
        # Step 1: Load the document
        documents = self.loader.load_document(file_path)
        
        # Step 2: Chunk the documents
        chunks = self.chunker.chunk_documents(documents)
        
        logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks")
        return chunks
    
    def ingest_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load and chunk multiple documents
        
        Args:
            file_paths: List of paths to document files
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.ingest_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                # Continue with other documents
                continue
        
        logger.info(f"Total chunks from all documents: {len(all_chunks)}")
        return all_chunks


# Example usage and testing
if __name__ == "__main__":
    # Example: Ingest a single document
    ingestor = DocumentIngestor(chunk_size=800, chunk_overlap=200)
    
    # Test with a sample text file
    sample_text = """
    This is a sample document for testing the ingestion pipeline.
    It contains multiple sentences and paragraphs to demonstrate
    how the chunking algorithm works with overlap between chunks.
    
    This is another paragraph to add more content to the document.
    We want to make sure that the chunking works correctly and that
    metadata is properly attached to each chunk.
    """
    
    # Save sample text to a file
    ensure_directory_exists('./data')
    with open('./data/sample.txt', 'w') as f:
        f.write(sample_text)
    
    # Ingest the sample document
    chunks = ingestor.ingest_document('./data/sample.txt')
    
    # Print results
    print(f"\nIngested {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}")
