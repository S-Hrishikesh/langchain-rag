# 📚 LangChain RAG Pipeline with ChromaDB & Groq

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent_Vector_Store-purple.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-orange.svg)](https://groq.com/)
[![Package Manager: uv](https://img.shields.io/badge/uv-Fast_Python_Packaging-yellow.svg)](https://github.com/astral-sh/uv)

A production-grade, modular Retrieval-Augmented Generation (RAG) pipeline built from scratch using **LangChain**, **ChromaDB**, **Sentence Transformers**, and blazing-fast LLM inference via **Groq Cloud**. 

Unlike standard RAG tutorials that rely on default settings and prone-to-fail assumptions, this project features custom-engineered classes designed to solve common vector math pitfalls, optimize memory usage via lazy loading, and guarantee collision-free multi-document indexing.

---

## ✨ Key Features

* ** Lazy-Loaded Embeddings (`Embedding_Manager`):** Wraps Hugging Face's `all-MiniLM-L6-v2` with lazy initialization—loading model weights into system memory only when embedding generation is explicitly triggered to reduce startup overhead and conserve RAM.
* ** True Cosine Similarity Math (`VectorStore` & `RAGRetriever`):** Explicitly overrides ChromaDB's default L2 (Squared Euclidean) distance metric by enforcing `"hnsw:space": "cosine"`. This ensures retriever similarity scores (`1 - distance`) mathematically map to true 0.0–1.0 percentages for reliable threshold filtering.
* ** Collision-Free Multi-PDF Indexing:** Automatically assigns unique UUID-based chunk hashes (`doc_{uuid}_{index}`) and rich metadata (chunk index, content length) during ingestion, allowing seamless appending of new documents without data overwriting or ID conflicts.
* ** Ultra-Fast Inference (`ChatGroq`):** Integrated with Groq's LPU architecture using state-of-the-art models like `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` for near-instantaneous synthesis of retrieved context.
* ** Modern Dependency Management:** Structured with `pyproject.toml` and locked via `uv` for deterministic, lightning-fast virtual environment setup.

---

## 🛠️ Architecture & Workflow

```text
[ PDF Documents ] -> [ Text Splitter ] -> [ Embedding_Manager ]
                                                  │ (all-MiniLM-L6-v2)
                                                  ▼
[ User Query ] ────> [ RAGRetriever ] <───> [ VectorStore ]
                            │               (ChromaDB / Cosine Space)
                            ▼
                     [ Context Filtering ]
                            │ (Score >= Threshold)
                            ▼
                   [ ChatGroq LLM ] ──────> [ Synthesized Response ]

langchain-rag/
│
├── data/
│   ├── pdf_files/             # Raw source PDF documents
│   └── vector_store/          # Persistent ChromaDB database files
│
├── notebook/
│   ├── document.ipynb         # Exploratory notebook for document processing
│   └── pdf_loader.ipynb       # Core RAG implementation & pipeline testing
│
├── main.py                    # Main execution script / API entry point
├── pyproject.toml             # Project dependencies & metadata
├── uv.lock                    # Deterministic lockfile for uv package manager
├── requirements.txt           # Standard pip dependency export
└── README.md                  # Project documentation and tutorial


