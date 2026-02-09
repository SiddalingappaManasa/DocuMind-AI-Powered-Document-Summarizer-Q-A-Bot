"""
app.py - Streamlit UI for RAG Document Q&A System

This is the main entry point for the application.
Provides a web interface for:
- Uploading documents (PDF, DOCX, TXT)
- Processing and indexing documents
- Asking questions
- Generating summaries
"""

import os
import streamlit as st
from pathlib import Path
import time

# Import our custom modules
from ingest import DocumentIngestor
from embeddings import EmbeddingManager
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline
from utils import get_config, logger, ensure_directory_exists, validate_file_type


# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """
    Initialize Streamlit session state variables
    """
    if 'config' not in st.session_state:
        try:
            st.session_state.config = get_config()
        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.stop()
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def initialize_system():
    """
    Initialize the RAG system components
    """
    config = st.session_state.config
    
    # Initialize RAG pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Initializing system..."):
            st.session_state.pipeline = RAGPipeline(
                api_key=config['openai_api_key'],
                embedding_model=config['embedding_model'],
                llm_model=config['llm_model'],
                temperature=config['temperature'],
                top_k=config['top_k_results']
            )
            
            # Initialize embedding manager
            embedding_manager = EmbeddingManager(
                api_key=config['openai_api_key'],
                model=config['embedding_model']
            )
            
            # Get embedding dimension
            dimension = embedding_manager.get_dimension()
            
            # Initialize vector store
            st.session_state.vector_store = VectorStoreManager(
                dimension=dimension,
                index_path=config['vector_db_path']
            )
            
            # Try to load existing index
            loaded = st.session_state.vector_store.initialize()
            
            if loaded:
                st.session_state.documents_processed = True
            
            # Connect vector store to pipeline
            st.session_state.pipeline.set_vector_store(st.session_state.vector_store)
            
            st.success("✅ System initialized successfully!")


def process_uploaded_files(uploaded_files):
    """
    Process uploaded documents and add them to the vector store
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    """
    config = st.session_state.config
    
    # Create data directory
    data_dir = "./data"
    ensure_directory_exists(data_dir)
    
    # Save uploaded files temporarily
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    # Process documents
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Ingest documents
        status_text.text("📄 Loading and chunking documents...")
        progress_bar.progress(20)
        
        ingestor = DocumentIngestor(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap']
        )
        chunks = ingestor.ingest_multiple_documents(file_paths)
        
        if not chunks:
            st.error("No content extracted from documents. Please check the files.")
            return
        
        # Step 2: Generate embeddings
        status_text.text(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        progress_bar.progress(50)
        
        embedding_manager = EmbeddingManager(
            api_key=config['openai_api_key'],
            model=config['embedding_model']
        )
        chunks_with_embeddings = embedding_manager.process_chunks(chunks)
        
        # Step 3: Add to vector store
        status_text.text("💾 Adding to vector database...")
        progress_bar.progress(80)
        
        st.session_state.vector_store.add_documents(chunks_with_embeddings)
        
        # Step 4: Save vector store
        st.session_state.vector_store.save()
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.session_state.documents_processed = True
        
        # Display statistics
        st.success(f"""
        **Processing Summary:**
        - Files processed: {len(uploaded_files)}
        - Total chunks: {len(chunks)}
        - Chunk size: {config['chunk_size']} tokens
        - Overlap: {config['chunk_overlap']} tokens
        """)
        
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        logger.error(f"Document processing error: {e}")


def render_sidebar():
    """
    Render the sidebar with document upload and settings
    """
    with st.sidebar:
        st.title("📚 Document Manager")
        
        # File upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Vector store information
        st.subheader("📊 System Status")
        
        if st.session_state.vector_store:
            info = st.session_state.vector_store.get_info()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", info['total_vectors'])
            with col2:
                st.metric("Dimensions", info['dimension'])
            
            if info['total_vectors'] > 0:
                st.success("✅ Ready for queries")
            else:
                st.warning("⚠️ No documents loaded")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        config = st.session_state.config
        
        st.info(f"""
        **Current Configuration:**
        - Model: {config['llm_model']}
        - Temperature: {config['temperature']}
        - Top-K: {config['top_k_results']}
        - Chunk Size: {config['chunk_size']}
        """)
        
        # Clear database button
        if st.button("🗑️ Clear Database", help="Remove all indexed documents"):
            if st.session_state.vector_store:
                st.session_state.vector_store.vector_store.clear()
                st.session_state.documents_processed = False
                st.session_state.chat_history = []
                st.rerun()


def render_qa_interface():
    """
    Render the Q&A interface
    """
    st.header("💬 Ask Questions")
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first!")
        return
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    with col2:
        if st.button("🧹 Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            # Get answer
            result = st.session_state.pipeline.answer_question(question)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'answer': result['answer'],
                'sources': result['sources'],
                'confidence': result['confidence']
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                
                # Answer
                st.markdown(f"**Answer:** {chat['answer']}")
                
                # Sources
                if chat['sources']:
                    with st.expander(f"📖 View {len(chat['sources'])} source(s)"):
                        for source in chat['sources']:
                            st.markdown(f"""
                            **Source {source['source_id']}:** {source['source_name']} (Page {source['page']})
                            
                            *Excerpt:* {source['text']}
                            
                            *Relevance Score:* {source['distance']:.4f}
                            """)
                            st.divider()
                
                st.divider()


def render_summary_interface():
    """
    Render the document summary interface
    """
    st.header("📝 Document Summary")
    
    if not st.session_state.documents_processed:
        st.warning("⚠️ Please upload and process documents first!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_length = st.select_slider(
            "Summary detail level:",
            options=["Brief (10 chunks)", "Medium (20 chunks)", "Detailed (30 chunks)"],
            value="Medium (20 chunks)"
        )
    
    with col2:
        generate_button = st.button("✨ Generate Summary", type="primary")
    
    if generate_button:
        # Parse max_chunks from selection
        max_chunks = int(summary_length.split("(")[1].split()[0])
        
        with st.spinner("Generating summary..."):
            result = st.session_state.pipeline.summarize_document(max_chunks=max_chunks)
            
            st.subheader("Summary")
            st.markdown(result['summary'])
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks Used", result['num_chunks_used'])
            with col2:
                st.metric("Sources", len(result.get('sources', [])))
            
            if result.get('sources'):
                with st.expander("📚 Source Documents"):
                    for source in result['sources']:
                        st.write(f"- {source}")


def main():
    """
    Main application entry point
    """
    # Title
    st.title("📚 RAG Document Q&A System")
    st.markdown("*Upload documents, ask questions, and get AI-powered answers with source citations*")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize system
    initialize_system()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📝 Summary", "ℹ️ About"])
    
    with tab1:
        render_qa_interface()
    
    with tab2:
        render_summary_interface()
    
    with tab3:
        st.header("About This System")
        st.markdown("""
        ### 🎯 Features
        - **Multi-format Support**: Upload PDF, DOCX, and TXT files
        - **Intelligent Chunking**: Documents are split into optimal chunks with overlap
        - **Semantic Search**: Uses OpenAI embeddings for accurate retrieval
        - **Source Citations**: Every answer includes references to source documents
        - **Summarization**: Generate comprehensive summaries of your documents
        
        ### 🔧 Technology Stack
        - **Frontend**: Streamlit
        - **Embeddings**: OpenAI text-embedding-3-small
        - **LLM**: GPT-4 Turbo
        - **Vector Store**: FAISS
        - **Document Processing**: PyPDF2, python-docx
        
        ### 📖 How to Use
        1. Upload documents using the sidebar
        2. Click "Process Documents" to index them
        3. Ask questions in the Q&A tab
        4. Generate summaries in the Summary tab
        
        ### 🔑 Configuration
        Set your OpenAI API key in a `.env` file:
        ```
        OPENAI_API_KEY=your_api_key_here
        ```
        
        ### 💡 Tips
        - Upload multiple related documents for better context
        - Ask specific questions for more accurate answers
        - Use the summary feature to quickly understand document content
        - Check source citations to verify information
        """)


if __name__ == "__main__":
    main()
