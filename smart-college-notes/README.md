\# Smart College Notes – Semantic Search + RAG



\## 📌 Overview

Smart College Notes is an AI-powered semantic search and Retrieval-Augmented Generation (RAG) system built using Endee as the base repository.



The system allows students to upload college notes (PDF) and ask natural language questions. The system retrieves relevant content using vector embeddings and generates intelligent answers using a local LLM (Llama3).



---



\## 🚀 Features

\- PDF ingestion and text extraction

\- Smart text chunking with overlap

\- Embedding generation using SentenceTransformers

\- Vector similarity search (cosine similarity)

\- Retrieval-Augmented Generation (RAG)

\- Local LLM integration via Ollama (Llama3)

\- Streamlit-based interactive UI



---



\## 🧠 System Architecture



1\. PDF → Text Extraction  

2\. Text → Chunking  

3\. Chunk → Embeddings  

4\. Store Embeddings (Vector Store)  

5\. Query → Embedding  

6\. Similarity Search  

7\. Retrieve Top-K Context  

8\. Context + Query → Llama3  

9\. Generated Answer  



---



\## 🛠 Tech Stack

\- Python

\- SentenceTransformers

\- Scikit-learn

\- Ollama (Llama3)

\- Streamlit

\- Endee (forked base repository)



---



\## ⚙️ Setup Instructions



\### 1. Clone repository
	https://github.com/Prakhar2784/endee.git





\### 2. Create virtual environment
	python -m venv venv

&nbsp;	venv\\Scripts\\activate





\### 3. Install dependencies

&nbsp;	pip install -r requirements.txt



\### 4. Install Ollama

Download from https://ollama.com



Pull model:ollama pull llama3



\### 5. Run ingestion

&nbsp;	python ingest.py



\### 6. Run application

&nbsp;	streamlit run app.py





---



\## 📊 Use Case

This project demonstrates how vector databases enable semantic search and how RAG systems enhance LLM responses using external knowledge sources.



---



\## 🎯 Why Endee?

Endee provides a vector database foundation for semantic search applications. This project builds on the forked Endee repository and demonstrates a real-world AI use case.



---



\## 📌 Future Improvements

\- Multi-PDF support

\- Persistent vector DB integration

\- Agentic workflows

\- Deployment to cloud



