# Chat with PDF

Live project link - https://chat-with-pdf1.streamlit.app/

A Retrieval-Augmented Generation (RAG) application built with Streamlit and OpenAI.

## Features

- Upload PDF documents
- Automatic text extraction
- Document chunking
- OpenAI embeddings generation
- Vector similarity search
- Top-K chunk retrieval
- Context-aware question answering

## Tech Stack

- Python
- Streamlit
- OpenAI API
- PyPDF
- Embeddings (text-embedding-3-small)

## How it Works

PDF
→ Text Extraction
→ Chunking
→ Embeddings
→ Similarity Search
→ Top-K Retrieval
→ GPT Response

## Run Locally

pip install -r requirements.txt

streamlit run app.py
