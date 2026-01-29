# 📄 RAG-Based Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **FAISS**, and **Mistral-7B**. This application allows you to upload documents and perform context-aware Q&A without high-cost infrastructure.

## ✨ Key Features
- **Multi-format Support**: Process both `.pdf` and `.txt` files seamlessly.
- **Asynchronous Ingestion**: Large documents are processed in the background using FastAPI `BackgroundTasks` to avoid blocking the API.
- **Vector Search**: Leverages **FAISS** for millisecond-latency similarity searching.
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` locally for high-quality, zero-cost vectorization.
- **Advanced LLM**: Integrated with **Mistral-7B-Instruct** for fact-grounded responses.
- **Safety**: Includes basic rate limiting to prevent API abuse.

## 🛠️ Architecture Decisions

### 1. Chunking Strategy
- **Size**: 700 characters (~150-200 tokens).
- **Overlap**: 100 characters.
- **Justification**: The embedding model has a context window of 256 tokens. 700 characters maximize information density while ensuring no text is truncated, while the overlap prevents losing context at the split points.

### 2. The Tech Stack
- **FastAPI**: Chosen for its native support for asynchronous tasks and Pydantic validation.
- **FAISS**: A lightweight alternative to heavy cloud vector databases, ideal for localized project deployments.
- **Mistral-7B via HF Hub**: Provides GPT-4 level reasoning for RAG tasks without the enterprise cost.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Hugging Face Access Token ([Get it here](https://huggingface.co/settings/tokens))

### Installation
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/yourusername/rag-system.git](https://github.com/yourusername/rag-system.git)
   cd rag-system
