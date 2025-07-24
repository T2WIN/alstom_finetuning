# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially based on projectBrief.md and other project-related information.

2025-07-24 08:51:53 - Log of updates made

## Project Goal
Build a document embedding pipeline that processes Word and Excel documents, extracts structured information, and stores it in a vector database for efficient retrieval and question-answering.

## Key Features
- Conversion of Word/Excel documents to Markdown and structured formats
- LLM-powered extraction of document structure, titles, and summaries
- Storage of document embeddings in Qdrant vector database
- Hard negative mining for training embedding models
- Asynchronous processing pipeline

## Overall Architecture
- Python-based system using asyncio for concurrency
- Modular design with processors for different file types
- Services for document conversion, LLM interaction, and vector storage
- Configuration management via central config.yaml
- Pydantic models for data validation stored in data_models.py

## Current Status
- Excel processing partially implemented (content sheet embedding missing)
- Word processing partially implemented (structure extraction complete)
- Hard negative mining feature added
- Missing: state management, query-answer pair generation, full Word pipeline, central config usage in all files, robust logging