# 🚀 QUICK START GUIDE

## Get Running in 5 Minutes!

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Key
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Run the App
```bash
streamlit run app.py
```

### Step 4: Use the System
1. **Upload** a document (PDF, DOCX, or TXT)
2. **Process** it (click "Process Documents")
3. **Ask** questions in the Q&A tab
4. **Generate** summaries in the Summary tab

---

## 📁 What's Included

### Core Files (Run These)
- `app.py` - Main Streamlit UI → **START HERE**
- `example_usage.py` - Programmatic examples

### Implementation Files
- `ingest.py` - Document loading & chunking
- `embeddings.py` - Embedding generation
- `vector_store.py` - FAISS database
- `rag_pipeline.py` - RAG workflow
- `utils.py` - Helper functions

### Documentation
- `README.md` - Full documentation
- `INSTALLATION.md` - Detailed setup
- `PROJECT_OVERVIEW.md` - Deep dive
- `PROJECT_STRUCTURE.md` - Architecture

---

## 🎯 Example Questions to Try

After uploading a document, try:
- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"
- "List the recommendations"

---

## 🔑 Configuration Options

Edit `.env` to customize:

```bash
# Models
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview

# Chunking
CHUNK_SIZE=800          # Tokens per chunk
CHUNK_OVERLAP=200       # Overlap between chunks

# Retrieval
TOP_K_RESULTS=5         # Number of chunks to retrieve

# Generation
TEMPERATURE=0.7         # LLM creativity (0-1)
```

---

## 💡 Tips

✅ Start with TXT files for testing
✅ Keep chunks at 500-1000 tokens
✅ Use overlap of 15-25% of chunk size
✅ Lower temperature (0.3) for factual Q&A
✅ Higher temperature (0.8) for creative tasks
✅ Increase top-K if answers lack context

---

## 🆘 Troubleshooting

**"OPENAI_API_KEY not found"**
→ Create `.env` file with your API key

**"No module named 'streamlit'"**
→ Run `pip install -r requirements.txt`

**"Vector store is empty"**
→ Upload and process documents first

**"Rate limit exceeded"**
→ Wait a few minutes or upgrade OpenAI plan

---

## 📚 Learning Path

1. **Start**: Run `streamlit run app.py`
2. **Explore**: Try different documents and questions
3. **Learn**: Read `PROJECT_OVERVIEW.md`
4. **Customize**: Edit configuration in `.env`
5. **Extend**: Modify code to add features

---

## 🎓 Next Steps

- [ ] Try with your own documents
- [ ] Experiment with different models
- [ ] Test various chunk sizes
- [ ] Add custom prompts
- [ ] Integrate with your workflow

---

**Need help?** Check the full README.md or open an issue!
