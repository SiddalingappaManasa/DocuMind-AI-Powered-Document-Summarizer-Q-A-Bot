"""
vector_store.py - FAISS vector database operations

This module handles:
- Creating and managing FAISS vector index
- Adding embeddings to the index
- Performing similarity search
- Persisting and loading the index
"""

import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from utils import logger, ensure_directory_exists


class FAISSVectorStore:
    """
    Manages a FAISS vector database for semantic search
    """
    
    def __init__(self, dimension: int, index_path: str = "./vector_db"):
        """
        Initialize the FAISS vector store
        
        Args:
            dimension: Dimension of the embedding vectors
            index_path: Directory path to save/load the index
        """
        self.dimension = dimension
        self.index_path = index_path
        
        # Create index directory if it doesn't exist
        ensure_directory_exists(index_path)
        
        # Initialize FAISS index (using L2 distance)
        # IndexFlatL2 is simple and accurate for small to medium datasets
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store metadata for each vector (chunk text and metadata)
        self.metadata_store = []
        
        # Track if index has been modified
        self.modified = False
        
        logger.info(f"Initialized FAISS vector store with dimension: {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add embeddings and their metadata to the vector store
        
        Args:
            embeddings: Numpy array of shape (n, dimension) containing embeddings
            metadata: List of metadata dicts for each embedding
        """
        # Validate inputs
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Ensure embeddings are float32 (required by FAISS)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata_store.extend(metadata)
        
        # Mark as modified
        self.modified = True
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store")
        logger.info(f"Total vectors in store: {self.index.ntotal}")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks with embeddings to the vector store
        
        Args:
            chunks: List of chunk dicts with 'embedding', 'text', and 'metadata' keys
        """
        # Extract embeddings and metadata
        embeddings = []
        metadata = []
        
        for chunk in chunks:
            if 'embedding' not in chunk:
                raise ValueError("Chunk must contain 'embedding' key")
            
            embeddings.append(chunk['embedding'])
            
            # Store both text and metadata
            metadata.append({
                'text': chunk['text'],
                'metadata': chunk['metadata']
            })
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to vector store
        self.add_embeddings(embeddings_array, metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for similar vectors in the store
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
        
        Returns:
            List of tuples (distance, metadata) for top-k results
        """
        # Ensure query is the right shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        # Check if index has vectors
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Limit top_k to available vectors
        k = min(top_k, self.index.ntotal)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            # Skip invalid indices
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            
            results.append((
                float(distance),
                self.metadata_store[idx]
            ))
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save(self, index_name: str = "faiss_index"):
        """
        Save the FAISS index and metadata to disk
        
        Args:
            index_name: Base name for the saved files
        """
        # Paths for index and metadata
        index_file = os.path.join(self.index_path, f"{index_name}.index")
        metadata_file = os.path.join(self.index_path, f"{index_name}_metadata.pkl")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            self.modified = False
            logger.info(f"Saved vector store to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load(self, index_name: str = "faiss_index") -> bool:
        """
        Load the FAISS index and metadata from disk
        
        Args:
            index_name: Base name for the files to load
        
        Returns:
            True if successful, False otherwise
        """
        index_file = os.path.join(self.index_path, f"{index_name}.index")
        metadata_file = os.path.join(self.index_path, f"{index_name}_metadata.pkl")
        
        # Check if files exist
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            logger.warning(f"Index files not found in {self.index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                self.metadata_store = pickle.load(f)
            
            self.modified = False
            logger.info(f"Loaded vector store with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """
        Clear all data from the vector store
        """
        self.index.reset()
        self.metadata_store = []
        self.modified = True
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dict with statistics
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata_store),
            'modified': self.modified
        }


class VectorStoreManager:
    """
    High-level interface for managing the vector store
    """
    
    def __init__(self, dimension: int, index_path: str = "./vector_db"):
        """
        Initialize the vector store manager
        
        Args:
            dimension: Dimension of embedding vectors
            index_path: Path to store the index
        """
        self.vector_store = FAISSVectorStore(dimension, index_path)
        self.index_loaded = False
    
    def initialize(self, force_new: bool = False) -> bool:
        """
        Initialize the vector store (load existing or create new)
        
        Args:
            force_new: If True, create a new index even if one exists
        
        Returns:
            True if index was loaded, False if new index was created
        """
        if force_new:
            logger.info("Creating new vector store")
            self.vector_store.clear()
            return False
        
        # Try to load existing index
        loaded = self.vector_store.load()
        self.index_loaded = loaded
        
        if loaded:
            logger.info("Loaded existing vector store")
        else:
            logger.info("No existing index found, will create new one")
        
        return loaded
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of chunks with embeddings
        """
        self.vector_store.add_chunks(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
        
        Returns:
            List of (distance, metadata) tuples
        """
        return self.vector_store.search(query_embedding, top_k)
    
    def save(self):
        """
        Save the vector store to disk
        """
        if self.vector_store.modified:
            self.vector_store.save()
            logger.info("Vector store saved")
        else:
            logger.info("Vector store unchanged, skipping save")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store
        
        Returns:
            Dict with statistics
        """
        stats = self.vector_store.get_stats()
        stats['index_loaded'] = self.index_loaded
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Create sample embeddings
    dimension = 1536  # OpenAI text-embedding-3-small dimension
    n_samples = 10
    
    # Generate random embeddings for testing
    sample_embeddings = np.random.rand(n_samples, dimension).astype(np.float32)
    
    # Create sample metadata
    sample_metadata = [
        {
            'text': f'This is sample text {i}',
            'metadata': {'source': 'test.txt', 'chunk_id': i}
        }
        for i in range(n_samples)
    ]
    
    # Initialize vector store
    store = FAISSVectorStore(dimension)
    
    # Add embeddings
    store.add_embeddings(sample_embeddings, sample_metadata)
    
    # Print stats
    stats = store.get_stats()
    print(f"\nVector store stats: {stats}")
    
    # Perform a search
    query_embedding = np.random.rand(dimension).astype(np.float32)
    results = store.search(query_embedding, top_k=3)
    
    print(f"\nSearch results (top 3):")
    for i, (distance, metadata) in enumerate(results):
        print(f"{i+1}. Distance: {distance:.4f}")
        print(f"   Text: {metadata['text']}")
        print(f"   Source: {metadata['metadata']['source']}\n")
    
    # Save and load test
    store.save("test_index")
    print("Saved vector store")
    
    # Create new store and load
    new_store = FAISSVectorStore(dimension)
    loaded = new_store.load("test_index")
    
    if loaded:
        print(f"Successfully loaded vector store")
        print(f"Loaded stats: {new_store.get_stats()}")
