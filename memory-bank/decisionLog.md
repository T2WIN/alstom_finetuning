# Decision Log

This file records architectural and implementation decisions using a list format.
2025-07-24 08:51:53 - Log of updates made.

## Decision
Use Qdrant as vector database

## Rationale 
- Open-source solution with self-hosting capability
- Support for multi-vector embeddings
- Good Python client library

## Implementation Details
- Created separate collections for documents and document rows
- Using sentence-transformers for embeddings
- Implemented deterministic UUIDs for consistent identification

## Decision
Implement hard negative mining

## Rationale
- Logical next step after query, answer pair creation.
- Creates more effective training data
- Addresses specific project requirements for QA systems

## Implementation Details
- Added HardNegativeProcessor class
- Uses sentence-transformers for embedding
- Mines hard negatives from existing dataset

## Decision
Async processing of files

## Rationale
- Faster processing via better use of computing resources

## Implementation Details
- Use of asyncio and async calls to LLM
- tqdm_gather in the main.py to show a progress bar and have a simple async architecture


## Decision
Updated logging configuration to use module-based component names

## Rationale 
- Aligns logging with actual module structure of the codebase
- Allows more granular control of log levels for specific modules
- Improves debugging and monitoring capabilities
- Follows standard Python logging best practices

## Implementation Details
- Modified [`document_embedding_pipeline/config.yaml`](document_embedding_pipeline/config.yaml) to use hierarchical logging configuration
- Set log levels for pipeline, services, and utils modules
- Preserved existing service-specific configurations
- Added comments to explain each configuration option
