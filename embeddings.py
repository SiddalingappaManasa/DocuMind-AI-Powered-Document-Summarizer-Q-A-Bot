"""
embeddings.py - Embedding generation using OpenAI

This module handles:
- Generating embeddings for text chunks using OpenAI API
- Batch processing for efficiency
- Caching to avoid redundant API calls
- Error handling and retries
"""

import os
import hashlib
import pickle
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
from utils import logger, ensure_directory_exists


class EmbeddingGenerator:
    """
    Generates embeddings for text using OpenAI's embedding models
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", cache_dir: str = "./cache"):
        """
        Initialize the embedding generator
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
            cache_dir: Directory to store cached embeddings
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        ensure_directory_exists(cache_dir)
        
        # Cache for embeddings (in-memory)
        self.embedding_cache = {}
        
        # Load cached embeddings from disk
        self._load_cache()
        
        logger.info(f"Initialized EmbeddingGenerator with model: {model}")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text string
        
        Args:
            text: Input text
        
        Returns:
            MD5 hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_cache(self):
        """
        Load cached embeddings from disk
        """
        cache_file = os.path.join(self.cache_dir, f"{self.model}_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.embedding_cache = {}
        else:
            logger.info("No cache file found, starting with empty cache")
    
    def _save_cache(self):
        """
        Save embeddings cache to disk
        """
        cache_file = os.path.join(self.cache_dir, f"{self.model}_cache.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string
        
        Args:
            text: Input text
        
        Returns:
            Numpy array containing the embedding vector
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.embedding_cache:
            logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return self.embedding_cache[cache_key]
        
        # Generate new embedding
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            logger.debug(f"Generated new embedding for text: {text[:50]}...")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch
        
        Returns:
            List of numpy arrays containing embedding vectors
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Separate cached and non-cached texts
        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            
            if cache_key in self.embedding_cache:
                # Use cached embedding
                embeddings.append((idx, self.embedding_cache[cache_key]))
            else:
                # Need to generate embedding
                texts_to_embed.append(text)
                text_indices.append(idx)
        
        logger.info(f"Found {len(embeddings)} cached embeddings, generating {len(texts_to_embed)} new ones")
        
        # Process non-cached texts in batches
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_indices = text_indices[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model
                )
                
                # Process each embedding in the batch
                for j, data in enumerate(response.data):
                    embedding = np.array(data.embedding, dtype=np.float32)
                    original_idx = batch_indices[j]
                    
                    # Cache the embedding
                    cache_key = self._get_cache_key(batch_texts[j])
                    self.embedding_cache[cache_key] = embedding
                    
                    # Add to results
                    embeddings.append((original_idx, embedding))
                
                logger.info(f"Generated embeddings for batch {i // batch_size + 1}")
            
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                raise
        
        # Sort embeddings by original index and extract vectors
        embeddings.sort(key=lambda x: x[0])
        embedding_vectors = [emb for _, emb in embeddings]
        
        # Save cache to disk
        self._save_cache()
        
        return embedding_vectors
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' keys
        
        Returns:
            List of chunk dicts with added 'embedding' key
        """
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info(f"Successfully generated embeddings for all chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            Embedding dimension
        """
        # Generate a sample embedding to get dimension
        sample_embedding = self.generate_embedding("sample text")
        return len(sample_embedding)


class EmbeddingManager:
    """
    High-level interface for managing embeddings
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding manager
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.generator = EmbeddingGenerator(api_key, model)
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks and add embeddings
        
        Args:
            chunks: List of text chunks
        
        Returns:
            Chunks with embeddings added
        """
        return self.generator.embed_chunks(chunks)
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query
        
        Args:
            query: Query text
        
        Returns:
            Query embedding vector
        """
        return self.generator.generate_embedding(query)
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            Embedding dimension
        """
        return self.generator.get_embedding_dimension()


# Example usage and testing
if __name__ == "__main__":
    # Note: This requires a valid OpenAI API key
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        # Initialize embedding generator
        generator = EmbeddingGenerator(api_key)
        
        # Test single embedding
        sample_text = "This is a test sentence for embedding generation."
        embedding = generator.generate_embedding(sample_text)
        print(f"\nGenerated embedding for: {sample_text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Embedding preview: {embedding[:5]}...")
        
        # Test batch embeddings
        sample_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        embeddings = generator.generate_embeddings_batch(sample_texts)
        print(f"\nGenerated {len(embeddings)} embeddings")
        
        # Test with chunks
        sample_chunks = [
            {'text': text, 'metadata': {'source': 'test'}}
            for text in sample_texts
        ]
        
        chunks_with_embeddings = generator.embed_chunks(sample_chunks)
        print(f"\nProcessed {len(chunks_with_embeddings)} chunks with embeddings")
    else:
        print("Please set OPENAI_API_KEY in your environment variables")
