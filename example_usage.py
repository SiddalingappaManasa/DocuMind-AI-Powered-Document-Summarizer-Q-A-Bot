"""
example_usage.py - Example usage of the RAG system components

This script demonstrates how to use the RAG system programmatically
without the Streamlit UI. Useful for:
- Integration with other systems
- Batch processing
- Testing
- Understanding the workflow
"""

import os
from dotenv import load_dotenv

# Import our modules
from ingest import DocumentIngestor
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from utils import ensure_directory_exists, logger


def example_basic_workflow():
    """
    Example: Basic RAG workflow from scratch
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic RAG Workflow")
    print("="*60 + "\n")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    # Step 1: Create sample document
    print("📝 Step 1: Creating sample document...")
    ensure_directory_exists('./data')
    
    sample_text = """
    Machine Learning is a subset of artificial intelligence that focuses on
    developing systems that can learn from and make decisions based on data.
    It involves training algorithms on large datasets to identify patterns
    and make predictions.
    
    There are three main types of machine learning:
    1. Supervised Learning: Learning from labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through trial and error
    
    Deep Learning is a specialized form of machine learning that uses
    neural networks with multiple layers. It has been particularly successful
    in areas like computer vision and natural language processing.
    """
    
    with open('./data/ml_basics.txt', 'w') as f:
        f.write(sample_text)
    
    print("✅ Sample document created: ./data/ml_basics.txt\n")
    
    # Step 2: Ingest and chunk the document
    print("📄 Step 2: Ingesting and chunking document...")
    ingestor = DocumentIngestor(chunk_size=800, chunk_overlap=200)
    chunks = ingestor.ingest_document('./data/ml_basics.txt')
    print(f"✅ Created {len(chunks)} chunks\n")
    
    # Step 3: Generate embeddings
    print("🔢 Step 3: Generating embeddings...")
    embedding_manager = EmbeddingManager(api_key, model='text-embedding-3-small')
    chunks_with_embeddings = embedding_manager.process_chunks(chunks)
    print(f"✅ Generated embeddings for {len(chunks_with_embeddings)} chunks\n")
    
    # Step 4: Create vector store
    print("💾 Step 4: Creating vector store...")
    dimension = embedding_manager.get_dimension()
    vector_store = VectorStoreManager(dimension, index_path='./vector_db')
    vector_store.initialize(force_new=True)
    vector_store.add_documents(chunks_with_embeddings)
    vector_store.save()
    print(f"✅ Vector store created with {vector_store.get_info()['total_vectors']} vectors\n")
    
    # Step 5: Initialize RAG pipeline
    print("🤖 Step 5: Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        api_key=api_key,
        embedding_model='text-embedding-3-small',
        llm_model='gpt-4-turbo-preview',
        temperature=0.7,
        top_k=3
    )
    pipeline.set_vector_store(vector_store)
    print("✅ RAG pipeline initialized\n")
    
    # Step 6: Ask questions
    print("💬 Step 6: Asking questions...\n")
    
    questions = [
        "What is machine learning?",
        "What are the main types of machine learning?",
        "What is deep learning?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        result = pipeline.answer_question(question)
        print(f"Answer: {result['answer']}\n")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {len(result['sources'])} chunk(s)")
        print("-" * 60 + "\n")


def example_document_summary():
    """
    Example: Generate document summary
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Document Summarization")
    print("="*60 + "\n")
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return
    
    # Load existing vector store
    embedding_manager = EmbeddingManager(api_key)
    dimension = embedding_manager.get_dimension()
    
    vector_store = VectorStoreManager(dimension, index_path='./vector_db')
    loaded = vector_store.initialize()
    
    if not loaded or vector_store.get_info()['total_vectors'] == 0:
        print("⚠️ No documents in vector store. Run example_basic_workflow() first.")
        return
    
    # Initialize pipeline
    pipeline = RAGPipeline(api_key)
    pipeline.set_vector_store(vector_store)
    
    # Generate summary
    print("📝 Generating summary...\n")
    result = pipeline.summarize_document(max_chunks=10)
    
    print("Summary:")
    print("-" * 60)
    print(result['summary'])
    print("-" * 60)
    print(f"\nChunks used: {result['num_chunks_used']}")
    print(f"Sources: {result['sources']}\n")


def example_batch_questions():
    """
    Example: Process multiple questions in batch
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Question Processing")
    print("="*60 + "\n")
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return
    
    # Load vector store
    embedding_manager = EmbeddingManager(api_key)
    dimension = embedding_manager.get_dimension()
    
    vector_store = VectorStoreManager(dimension, index_path='./vector_db')
    loaded = vector_store.initialize()
    
    if not loaded or vector_store.get_info()['total_vectors'] == 0:
        print("⚠️ No documents in vector store. Run example_basic_workflow() first.")
        return
    
    # Initialize pipeline
    pipeline = RAGPipeline(api_key, temperature=0.5)  # Lower temperature for more focused answers
    pipeline.set_vector_store(vector_store)
    
    # Batch of questions
    questions = [
        "Define machine learning in one sentence",
        "List the three types of machine learning",
        "How is deep learning different?",
        "What are neural networks?"
    ]
    
    results = []
    
    print("Processing batch of questions...\n")
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        result = pipeline.answer_question(question)
        results.append({
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence']
        })
        print(f"✅ {result['confidence']} confidence\n")
    
    # Summary
    print("\n" + "="*60)
    print("BATCH RESULTS SUMMARY")
    print("="*60 + "\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. Q: {r['question']}")
        print(f"   A: {r['answer'][:100]}...")
        print(f"   Confidence: {r['confidence']}\n")


def example_custom_parameters():
    """
    Example: Using custom parameters for specialized use cases
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Parameters")
    print("="*60 + "\n")
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return
    
    # Example 1: High-precision retrieval (more chunks, lower temperature)
    print("Configuration 1: High Precision")
    print("-" * 40)
    
    pipeline_precise = RAGPipeline(
        api_key=api_key,
        llm_model='gpt-4-turbo-preview',
        temperature=0.3,  # Lower temperature = more focused
        top_k=7  # More context
    )
    
    print(f"Temperature: {pipeline_precise.temperature}")
    print(f"Top-K: {pipeline_precise.top_k}")
    print("Use case: Technical documentation, legal documents\n")
    
    # Example 2: Creative responses (higher temperature)
    print("Configuration 2: Creative")
    print("-" * 40)
    
    pipeline_creative = RAGPipeline(
        api_key=api_key,
        llm_model='gpt-4-turbo-preview',
        temperature=0.9,  # Higher temperature = more creative
        top_k=3  # Less context, more freedom
    )
    
    print(f"Temperature: {pipeline_creative.temperature}")
    print(f"Top-K: {pipeline_creative.top_k}")
    print("Use case: Creative writing, brainstorming\n")
    
    # Example 3: Fast and economical (smaller model)
    print("Configuration 3: Fast & Economical")
    print("-" * 40)
    
    pipeline_fast = RAGPipeline(
        api_key=api_key,
        llm_model='gpt-3.5-turbo',  # Faster, cheaper model
        temperature=0.5,
        top_k=3
    )
    
    print(f"Model: {pipeline_fast.llm_model}")
    print(f"Temperature: {pipeline_fast.temperature}")
    print("Use case: High-volume queries, simple Q&A\n")


def main():
    """
    Main function to run all examples
    """
    print("\n🚀 RAG System - Example Usage Scripts\n")
    
    # Check for API key
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: Please set OPENAI_API_KEY in your .env file")
        print("See .env.example for reference")
        return
    
    # Menu
    print("Choose an example to run:")
    print("1. Basic RAG Workflow (recommended for first run)")
    print("2. Document Summarization")
    print("3. Batch Question Processing")
    print("4. Custom Parameters Examples")
    print("5. Run All Examples")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == '1':
        example_basic_workflow()
    elif choice == '2':
        example_document_summary()
    elif choice == '3':
        example_batch_questions()
    elif choice == '4':
        example_custom_parameters()
    elif choice == '5':
        example_basic_workflow()
        example_document_summary()
        example_batch_questions()
        example_custom_parameters()
    elif choice == '0':
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice. Please run again.")
    
    print("\n✅ Done! Check out app.py for the Streamlit UI version.\n")


if __name__ == "__main__":
    main()
