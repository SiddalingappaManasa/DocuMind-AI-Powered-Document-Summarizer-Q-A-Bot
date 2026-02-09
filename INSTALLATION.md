# 📦 Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Step-by-Step Installation

### 1. Check Python Version

```bash
python --version
# Should show Python 3.8 or higher
```

If you need to install Python:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3`
- **Linux**: `sudo apt-get install python3 python3-pip`

### 2. Create Project Directory

```bash
# Create and navigate to project directory
mkdir rag_document_qa
cd rag_document_qa
```

### 3. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- Streamlit (Web UI)
- OpenAI (API client)
- FAISS (Vector database)
- PyPDF2 (PDF processing)
- python-docx (DOCX processing)
- And other dependencies...

### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

Add your API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 6. Verify Installation

```bash
# Test imports
python -c "import streamlit; import openai; import faiss; print('✅ All dependencies installed successfully!')"
```

### 7. Run the Application

```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`

## 🔧 Troubleshooting Installation

### Issue: pip not found

```bash
# Install pip
python -m ensurepip --upgrade
```

### Issue: FAISS installation fails on Windows

```bash
# Use conda instead
conda install -c pytorch faiss-cpu
```

### Issue: Permission denied

```bash
# Use --user flag
pip install --user -r requirements.txt
```

### Issue: Old pip version

```bash
# Upgrade pip
pip install --upgrade pip
```

### Issue: SSL Certificate errors

```bash
# Use trusted host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## 🖥️ Platform-Specific Notes

### Windows

- Make sure to use `python` instead of `python3`
- Use backslashes in paths: `data\sample.txt`
- Activate venv: `venv\Scripts\activate`

### macOS

- May need to use `python3` and `pip3`
- If you get SSL errors, try: `pip install --upgrade certifi`
- Activate venv: `source venv/bin/activate`

### Linux

- May need sudo for system-wide installation
- Prefer virtual environments
- Install python3-dev if needed: `sudo apt-get install python3-dev`

## 📊 Verify System Requirements

```bash
# Check disk space (need ~500MB for dependencies)
df -h

# Check memory (recommend 4GB+)
free -h

# Check CPU
lscpu
```

## 🚀 Quick Start Test

After installation, test with this workflow:

1. Start the app: `streamlit run app.py`
2. Upload a sample TXT file
3. Process it
4. Ask: "What is this document about?"
5. Generate a summary

## 💡 Development Setup

For development work:

```bash
# Install additional dev dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

## 🔄 Updating

To update dependencies:

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Or update specific package
pip install --upgrade openai
```

## 🆘 Getting Help

If you encounter issues:

1. Check the [README.md](README.md) troubleshooting section
2. Verify your Python version: `python --version`
3. Verify your pip version: `pip --version`
4. Check OpenAI API status: https://status.openai.com/
5. Review logs in the terminal
6. Open an issue on GitHub

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] `.env` file created with OpenAI API key
- [ ] Application runs without errors
- [ ] Can upload and process documents
- [ ] Can ask questions and get answers

---

**You're all set! 🎉**

Proceed to the [README.md](README.md) for usage instructions.
