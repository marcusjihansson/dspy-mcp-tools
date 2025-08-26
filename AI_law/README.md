# Database files (Postgresql)

This folder contains source PDFs for the Regulatory Analyzer’s RAG pipeline. 

Ingest these documents into the PostgreSQL ai_law_db (table rag_documents) and generate vector embeddings (e.g., Ollama bge-m3:latest) as described in postgresql.md. 

Files are reference materials only; large files should use Git LFS. Do not store secrets here. For setup and ingestion steps, see postgresql.md in the repository root.
