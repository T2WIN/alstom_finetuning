# System Patterns

This file documents recurring patterns and standards used in the project.
2025-07-24 08:51:53 - Log of updates made.

## Coding Patterns
- Pydantic models for data validation
- Abstract Base Classes (ABCs) for processor interfaces
- Asynchronous processing using asyncio
- Centralized configuration management

## Architectural Patterns
- Modular service architecture (docling, LLM, Qdrant, unoserver)
- Multi-vector storage in Qdrant
- Processor pipeline pattern for document handling
- Separation of concerns between parsing, processing and storage

## Testing Patterns
- Unit tests for services and utilities
- Integration tests for pipeline components
- Mocking for external dependencies