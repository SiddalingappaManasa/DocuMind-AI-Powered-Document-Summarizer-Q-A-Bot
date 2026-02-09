# 📚 RAG Document Summarizer & Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system for document question-answering and summarization, built with Streamlit, OpenAI, and FAISS.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)

## 🎯 Features

- ✅ **Multi-format Document Support**: Upload PDF, DOCX, and TXT files
- ✅ **Intelligent Text Chunking**: Splits documents into 500-1000 token segments with overlap
- ✅ **Semantic Search**: Uses OpenAI embeddings for accurate retrieval
- ✅ **Question Answering**: Get precise answers with source citations
- ✅ **Document Summarization**: Generate comprehensive summaries
- ✅ **FAISS Vector Store**: Fast and efficient similarity search
- ✅ **Caching**: Reduces API calls and improves performance
- ✅ **Clean UI**: User-friendly Streamlit interface
- ✅ **Production-Ready**: Error handling, logging, and modular architecture

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Top-K Retrieval → LLM → Answer + Citations
                ↓                              ↓
         OpenAI API                      FAISS Index
```

## 📁 Project Structure

```
rag_document_qa/
│
├── app.py                 # Streamlit UI (main entry point)
├── ingest.py             # Document loading & chunking
├── embeddings.py         # Embedding generation with caching
├── vector_store.py       # FAISS vector database operations
├── rag_pipeline.py       # RAG retrieval + LLM generation
├── utils.py              # Helper functions & configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Example environment variables
├── README.md            # This file
│
├── data/                # Directory for uploaded documents
└── vector_db/          # Directory for FAISS index storage
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag_document_qa
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Uploading Documents

1. Click on the **sidebar** to access the Document Manager
2. Click **"Browse files"** and select PDF, DOCX, or TXT files
3. Click **"Process Documents"** to index them
4. Wait for the processing to complete (progress bar shown)

### Asking Questions

1. Go to the **Q&A tab**
2. Type your question in the input box
3. Click **"Ask"**
4. View the answer with source citations
5. Expand sources to see relevant excerpts

### Generating Summaries

1. Go to the **Summary tab**
2. Select the detail level (Brief, Medium, or Detailed)
3. Click **"Generate Summary"**
4. View the comprehensive summary

## 🔧 Configuration

All configuration is managed through environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional (with defaults)
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview
TEMPERATURE=0.7
VECTOR_DB_PATH=./vector_db
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required** |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | OpenAI LLM model | `gpt-4-turbo-preview` |
| `TEMPERATURE` | LLM creativity (0-1) | `0.7` |
| `CHUNK_SIZE` | Target chunk size in tokens | `800` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `TOP_K_RESULTS` | Number of chunks to retrieve | `5` |

## 📚 Module Documentation

### app.py
**Main Streamlit application**

- Handles UI rendering
- Manages file uploads
- Orchestrates document processing
- Displays Q&A and summary interfaces

### ingest.py
**Document loading and chunking**

Key classes:
- `DocumentLoader`: Loads PDF, DOCX, TXT files
- `TextChunker`: Splits text into overlapping chunks
- `DocumentIngestor`: Orchestrates loading and chunking

### embeddings.py
**Embedding generation**

Key classes:
- `EmbeddingGenerator`: Generates embeddings using OpenAI
- `EmbeddingManager`: High-level embedding interface

Features:
- Batch processing for efficiency
- In-memory and disk caching
- Automatic cache management

### vector_store.py
**FAISS vector database**

Key classes:
- `FAISSVectorStore`: Core FAISS operations
- `VectorStoreManager`: High-level vector store interface

Features:
- L2 distance similarity search
- Index persistence (save/load)
- Metadata management

### rag_pipeline.py
**RAG orchestration**

Key class:
- `RAGPipeline`: Main RAG workflow

Features:
- Query embedding generation
- Context retrieval
- Prompt engineering
- Answer generation with citations

### utils.py
**Helper functions**

Functions:
- `get_config()`: Load configuration
- `clean_text()`: Text preprocessing
- `count_tokens()`: Token counting
- `format_source_citation()`: Citation formatting

## 🎓 How It Works

### 1. Document Ingestion
```python
# Load document
documents = loader.load_document("sample.pdf")

# Chunk into smaller segments
chunks = chunker.chunk_documents(documents)
# Result: List of 800-token chunks with 200-token overlap
```

### 2. Embedding Generation
```python
# Generate embeddings for chunks
chunks_with_embeddings = embedding_manager.process_chunks(chunks)
# Each chunk now has a 1536-dimensional vector
```

### 3. Vector Storage
```python
# Add to FAISS index
vector_store.add_documents(chunks_with_embeddings)

# Save for later use
vector_store.save()
```

### 4. Query Processing
```python
# User asks a question
query = "What is machine learning?"

# Convert query to embedding
query_embedding = embedding_manager.get_query_embedding(query)

# Find similar chunks
results = vector_store.search(query_embedding, top_k=5)
```

### 5. Answer Generation
```python
# Create prompt with retrieved context
prompt = create_qa_prompt(query, retrieved_chunks)

# Generate answer using LLM
answer = llm.generate(prompt)

# Return with source citations
```

## 🧪 Testing

### Test Individual Modules

Each module can be tested independently:

```bash
# Test document ingestion
python ingest.py

# Test embedding generation
python embeddings.py

# Test vector store
python vector_store.py

# Test RAG pipeline
python rag_pipeline.py
```

### Integration Test

1. Start the app: `streamlit run app.py`
2. Upload a sample document
3. Ask a test question
4. Verify the answer and citations

## 📊 Performance Optimization

### Caching Strategies

1. **Embedding Cache**: Stores generated embeddings to avoid redundant API calls
2. **FAISS Index**: Persists vector store to disk for fast loading
3. **Streamlit Cache**: Uses `@st.cache_data` for expensive operations

### Best Practices

- Use smaller chunk sizes for precise answers
- Use larger chunk sizes for broader context
- Adjust `top_k` based on your use case (3-10 recommended)
- Use `text-embedding-3-small` for cost efficiency
- Use `gpt-4-turbo` for best quality answers

## 🔒 Security

- API keys stored in `.env` file (never commit to git)
- Input validation for uploaded files
- Error handling for malformed documents
- Secure file storage in isolated directories

## 🐛 Troubleshooting

### Common Issues

**Issue**: "OPENAI_API_KEY not found"
```bash
# Solution: Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-key" > .env
```

**Issue**: "No module named 'streamlit'"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: "Vector store is empty"
```bash
# Solution: Upload and process documents first
```

**Issue**: PDF extraction fails
```bash
# Solution: Some PDFs are image-based and need OCR
# Use tools like Tesseract for OCR preprocessing
```

## 🚧 Limitations

- Maximum file size depends on available memory
- Image-based PDFs require OCR preprocessing
- Complex tables may not be extracted perfectly
- Answer quality depends on document quality and relevance

## 🔮 Future Enhancements

- [ ] Support for more file formats (CSV, JSON, HTML)
- [ ] Advanced chunking strategies (semantic chunking)
- [ ] Multi-language support
- [ ] User authentication
- [ ] Conversation history persistence
- [ ] Advanced analytics dashboard
- [ ] OCR support for scanned documents
- [ ] Hybrid search (keyword + semantic)

## 📝 Code Quality

- **Modular Design**: Each component is independent and testable
- **Clear Comments**: Every function documented with docstrings
- **Error Handling**: Comprehensive try-catch blocks
- **Logging**: Detailed logging for debugging
- **Type Hints**: Python type annotations throughout
- **Clean Code**: PEP 8 compliant

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure code quality (linting, type checking)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for embeddings and LLM APIs
- Meta AI for FAISS
- Streamlit for the amazing UI framework
- The open-source community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the ML community**
