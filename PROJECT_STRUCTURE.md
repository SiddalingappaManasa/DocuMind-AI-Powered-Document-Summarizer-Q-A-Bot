# RAG Document Summarizer & Q&A System

## Project Structure

```
rag_document_qa/
│
├── app.py                  # Streamlit UI - main entry point
├── ingest.py              # Document loading and text chunking
├── embeddings.py          # Embedding generation using OpenAI
├── vector_store.py        # FAISS vector database operations
├── rag_pipeline.py        # RAG retrieval + LLM generation pipeline
├── utils.py               # Helper functions and utilities
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── README.md             # Project documentation
│
├── data/                 # Directory for uploaded documents
│   └── .gitkeep
│
└── vector_db/           # Directory for FAISS index storage
    └── .gitkeep
```

## File Descriptions

### 1. **app.py**
- Streamlit web interface
- Handles file uploads (PDF, DOCX, TXT)
- Provides Q&A and summarization modes
- Displays answers with source citations

### 2. **ingest.py**
- Loads documents from various formats
- Extracts text from PDF, DOCX, TXT files
- Cleans and preprocesses text
- Chunks text into 500-1000 token segments with overlap

### 3. **embeddings.py**
- Generates embeddings using OpenAI's embedding models
- Handles batch processing for efficiency
- Implements caching to avoid redundant API calls

### 4. **vector_store.py**
- Manages FAISS vector database
- Stores and retrieves document embeddings
- Performs similarity search for top-k relevant chunks
- Handles index persistence (save/load)

### 5. **rag_pipeline.py**
- Orchestrates the RAG workflow
- Retrieves relevant chunks based on query
- Constructs prompts for LLM
- Generates answers and summaries using OpenAI GPT
- Returns responses with source citations

### 6. **utils.py**
- Helper functions (token counting, text cleaning)
- Configuration management
- Error handling utilities
- Logging setup

## Key Features

✅ Multi-format document support (PDF, DOCX, TXT)
✅ Intelligent text chunking with overlap
✅ OpenAI embeddings for semantic search
✅ FAISS vector store for fast retrieval
✅ Question answering with source citations
✅ Document summarization
✅ Streamlit web interface
✅ Caching for improved performance
✅ Environment variable management
✅ Production-ready error handling
