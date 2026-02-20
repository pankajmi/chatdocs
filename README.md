# 📄 ChatDocs: Local PDF Q&A

**ChatDocs** is a CLI tool that allows you to chat with your PDF documents locally. It uses **RAG (Retrieval-Augmented Generation)** to find answers within your documents without sending your data to the cloud.

## 🚀 Features

- **Fully Local**: No API keys required. Runs entirely on your hardware via Ollama.
- **Context Aware**: Uses a "History-Aware Retriever" to understand follow-up questions.
- **Smart Chunking**: Utilizes recursive character splitting for better context retention.
- **Lightweight UI**: Simple CLI interface with a loading animation during processing.


## 📋 Prerequisites

Before running the application, ensure you have the following installed:
1. Python 3.9+
2. [Ollama](https://ollama.com/): Download and install Ollama
3. [uv](https://docs.astral.sh/uv/getting-started/installation/): python package manager
4. Required Models:
Open your terminal and pull the necessary models:
Ensure you have [Ollama](https://ollama.com/) installed and running. Then, pull the required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```
## 🛠️ Setup
### 1. Clone & Install

```bash
git clone https://github.com/your-username/chatdocs.git
cd chatdocs
```
### 2. Install dependencies

```bash
uv sync
```

### 3. Run
```bash
python main.py documents/resume.pdf "What is the candidate's experience with Python?"
```

## 🧠 How It Works
1. **Ingestion**: The PyPDFLoader reads the document.
2. **Chunking**: The text is broken into 1,000-character segments with overlap to prevent losing context at boundaries.
3. **Embedding**: `nomic-embed-text` converts these chunks into mathematical vectors.
4. **Vector Store**: Vectors are stored in a local Chroma database.
5. **Retrieval**: When you ask a question, the system finds the most relevant chunks.
6. **Generation**: Llama3 uses the retrieved chunks to formulate a concise answer.
