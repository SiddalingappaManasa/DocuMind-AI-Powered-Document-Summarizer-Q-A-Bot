# DocuMind – RAG-Based Document Summarizer & Context-Aware Q&A System

DocuMind is an AI-powered document intelligence system that allows users to upload documents (PDF, DOCX, TXT) and generate concise summaries and context-aware answers.

The system is designed to handle long documents that exceed LLM context limits by combining semantic retrieval with language model reasoning, ensuring responses are grounded in the source content rather than generic model outputs.

This project applies Retrieval-Augmented Generation (RAG) principles to improve reliability and factual grounding in document-based question answering.
🚀 Features

✅ Upload documents (PDF, Word, Text)
✅ AI-generated summaries in simple language
✅ Ask natural language questions about the document
✅ Retrieval-augmented approach combining semantic search and language model reasoning for factual and contextual response
✅ User-friendly Streamlit web interface

🛠️ Tech Stack

Python
LangChain (for document processing & retrieval)
FAISS (vector database for semantic search)
OpenAI API (for natural language understanding & generation)
Streamlit (for interactive UI)

📂 Project Structure
├── app.py             # Main Streamlit app  
├── requirements.txt   # Dependencies  
├── sample_docs/       # Example documents to test  
└── README.md          # Project documentation 

🧠Key Design Decisions

-Used a Retrieval-Augmented Generation (RAG) approach to ground LLM responses in document content and reduce hallucinations instead of relying on direct prompting.
-Applied overlapping chunking while splitting documents to preserve contextual continuity across sections and improve semantic retrieval quality.
-Used FAISS as the vector database to enable efficient and scalable similarity search over document embeddings.
-Designed constrained prompts so the language model answers strictly based on retrieved document chunks, improving factual accuracy and reliability.

📈 Evaluation

DocuMind was evaluated qualitatively by comparing its responses against direct LLM prompting on the same documents.

- Responses generated using the RAG pipeline were more contextually accurate and relevant.
- Hallucinated or unsupported answers were significantly reduced when answers were grounded in retrieved document chunks.
- Summaries remained concise while retaining key factual information from the source documents.

This evaluation highlights the effectiveness of retrieval-augmented generation for document intelligence tasks.

🧠 NLP & System Design

- Performed text extraction and preprocessing (cleaning, tokenization, chunking)
- Split long documents into overlapping chunks to preserve context
- Converted text chunks into embeddings for semantic similarity search
- Retrieved top relevant chunks using FAISS
- Passed retrieved context to the language model for accurate, context-aware answers

⚡ How to Run Locally

Clone the repo:

git clone https://github.com/your-username/DocuMind.git
cd DocuMind


Create virtual environment & install requirements:

pip install -r requirements.txt


Add your OpenAI API key:

export OPENAI_API_KEY="your_api_key_here"   # For Mac/Linux
setx OPENAI_API_KEY "your_api_key_here"    # For Windows


Run the app:

streamlit run app.py

🎯 Use Cases

📚 Education → Summarize research papers & notes

🏥 Healthcare → Quick access to patient records

⚖️ Law → Summarize lengthy case files

🏢 Corporate → HR policies, project reports

🏛️ Governance → Public schemes & policies

🌟 Future Enhancements

Support for multiple document uploads

Multi-language support

Export summarized reports as PDF/Word

Fine-tuned models for domain-specific tasks

🤝 Contribution

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to improve.

📧 Contact

👩‍💻 Developed by Manasa Siddalingappa
🔗 LinkedIn
 | Portfolio
