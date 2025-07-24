# Project Structure

### Raw structure
- .env
- .gitignore
- .rooignore
- document_embedding_pipeline
  - README.md
  - _temp
  - commit_history.txt
  - config.yaml
  - data
    - test
  - excel_processing.log
  - hard_negative_feature.md
  - local_qdrant_db
    - .lock
    - collection
      - document_rows
        - storage.sqlite
      - documents
        - storage.sqlite
      - engineering_document_embeddings
        - storage.sqlite
    - meta.json
  - logs
    - pipeline.log
    - pipeline.log.1
  - main.py
  - models
    - EasyOcr
    - ds4sd--CodeFormula
    - ds4sd--DocumentFigureClassifier
    - ds4sd--SmolDocling-256M-preview
    - ds4sd--docling-models
  - prompt_for_building_features.md
  - pyproject.toml
  - qdrant_db
    - .lock
    - collection
      - document_collection
        - storage.sqlite
    - meta.json
  - requirements.txt
  - spec.md
  - src
    - __init__.py
    - data_models.py
    - document_embedding_pipeline.egg-info
      - PKG-INFO
      - SOURCES.txt
      - dependency_links.txt
      - top_level.txt
    - pipeline
      - __init__.py
      - base_processor.py
      - excel_processor.py
      - hard_negative_processor.py
      - word_processor.py
    - prompts.py
    - scripts
      - process_excel_demo.py
      - test_code.py
      - walk_word_pipeline.py
    - services
      - __init__.py
      - docling_service.py
      - llm_service.py
      - qdrant_service.py
      - unoserver_service.py
    - tests
      - __init__.py
      - test_pipeline
        - __init__.py
        - test_accept_changes.py
        - test_word_processor.py
      - test_services
        - __init__.py
        - test_llm_service.py
        - test_qdrant_service.py
        - test_unoserver_service.py
      - test_utils
        - __init__.py
        - test_logging_setup.py
    - utils
      - __init__.py
      - __pycache__
      - config_loader.py
      - file_handler.py
      - logging_setup.py
      - state_manager.py
      - table_parsing.py
  - verify_out
- projectBrief.md


### Explanations of code :

#### docling_service.py
This file defines a DoclingService class which serves as a wrapper for the docling library. Its primary role is to handle the conversion of documents into Markdown format. The service is configured to process PDF inputs and includes error logging to capture any failures during the conversion process.

#### llm_service.py
This file defines the LLMService, which acts as a centralized client for interacting with LLMs hosted on a local Ollama server. It uses the instructor library to ensure that LLM outputs conform to specified Pydantic data models, effectively producing structured JSON. The service is responsible for truncating input prompts that exceed a configured token limit and handles all API communication and error logging. It loads its configuration, such as the Ollama URL and tokenizer model, from an external file.

#### qdrant_service.py
Upon initialization, the service loads a local Sentence Transformer model to generate embeddings and ensures two Qdrant collections exist: one for parent documents (documents) and another for individual table rows (document_rows). It provides distinct methods for upserting Word and Excel data; for Word files, it embeds section summaries into a single multi-vector record, while for Excel files, it embeds both table summaries into the document record and each individual data row into the separate rows collection. The service uses deterministic UUIDs to ensure consistent record identification across runs.

#### unoserver_service.py
The function executes unoconvert as a subprocess to convert a document from an input path to a specified file type at an output path. It includes error handling for cases where the conversion fails or the unoconvert command is not found in the system's PATH, logging any issues before raising an exception. This module is essential for file format transformations, such as converting a .docx file to a .pdf.

#### data_models.py
It establishes clear schemas for both Word and Excel documents. For Excel, it differentiates between ContentSheet and TableSheet and includes models for table headers and summaries. For Word documents, it uses a recursive Section model to represent the document's hierarchical structure. These models are crucial for validating the outputs from parsing stages and LLM interactions throughout the data processing pipeline.

#### prompt.py
It defines specific string templates for various extraction and summarization tasks. These include prompts for extracting a document's title and hierarchical structure simultaneously, summarizing individual text sections and tables, and generating a global document summary from the individual section summaries. Each prompt is designed with placeholders to be dynamically filled with document content before being sent to the LLM.

#### base_processor.py
It uses Python's abc module to establish a contract that all subclasses must follow. Specifically, any class that inherits from BaseProcessor is required to implement its own __init__ method and an asynchronous process method, ensuring a consistent interface for handling different types of files.

#### excel_processor.py
Inheriting from BaseProcessor, this class first converts older .xls files to the .xlsx format. It then uses a factory pattern to parse the workbook, classifying each sheet and leveraging the LLMService to extract a global title and analyze complex tables. Finally, the processor formats all the extracted and generated data into a structured ExcelDocumentPayload model, preparing it for storage and downstream use.

#### word_processor.py
The processor manages the conversion of Word documents to Markdown (using an intermediate PDF step) and then employs the LLMService to asynchronously extract the document's hierarchical structure and title. However, the implementation is currently incomplete, as major parts of the logic—including section summarization, table summarization, global summary generation, and the final creation of the data payload—are commented out. A function to accept tracked changes is also present but not yet implemented.

#### hard_negative_processor.py
This processor takes a dataset of (query, answer) pairs, embeds them using a Sentence Transformer model, and then "mines" for hard negatives. For each query, it finds incorrect answers from the dataset that are semantically very similar, creating training triplets of (query, positive_answer, hard_negative_answer). This entire functionality is a new addition and was not part of the original project specification.

#### config_loader.py
This class uses a singleton pattern to load settings from a config.yaml file just once. It provides a convenient get method that allows other parts of the application to access nested configuration values using a dot-separated string (e.g., llm.max_input_tokens). This approach centralizes all configuration management into a single, easy-to-use module.

#### logging_setup.py
The function sets up a root logger with two distinct handlers: a console handler that prints logs to the standard output with a configurable severity level, and a RotatingFileHandler that saves more detailed DEBUG-level logs to a pipeline.log file. It defines different formatters for each handler, ensuring console logs are concise while file logs are comprehensive for debugging purposes. This module centralizes the logging configuration for the entire project.

#### table_parsing.py 
It uses a Depth-First Search (DFS) algorithm to find contiguous cell blocks and classify worksheets as either "content" or "table" sheets. The ParsedTableSheet class implements advanced logic to heuristically find table boundaries within a sheet and extract structured data from them. It then leverages the LLMService to generate a natural language summary for each identified table, using data from other "content" sheets as additional context.

#### main.py
The script sets up configuration and logging, discovers Word and Excel files from a hardcoded directory, and then instantiates the appropriate processors. It uses asyncio to run the processing tasks concurrently for a subset of the discovered files, with a progress bar provided by tqdm. Key functionalities specified in the original document, such as command-line argument handling, the state management system (state.json), and saving results to the Qdrant database, are currently commented out or absent from the implementation. 


# Missing features from original spec

### Central config usage in all files

Explanation : All files should import their config values from the central config.yaml file.
Reason why it isn't implemented : Not enough time spent on it.
Do we want to implement it ? : Yes

### State management system

Explanation : Using a central state.json file to track progress, retrying failed documents and managing temporary files.
Reason why it isn't implemented : Time consuming to develop.
Do we want to implement it : Yes

### Full Word document pipeline

Explanation : Converts doc and docx documents into a markdown representation with llm extracted metadata. Accepting changes automatically.
Currently implemented doc to docx conversion and markdown + simple structure extraction.
Need to keep working on it to add summarization and embedding. It is my current priority.
Reason why it isn't implemented : I am working on it, it is just not finished yet. Accepting changes is very difficult on Linux.

### Word pipeline extraction logic

Explanation : Split word document handling between small (<=5k tokens) and large documents (> 5k tokens). Use LLM structure extraction only for smaller docs.
Reason why it isn't implemented : I am working on it.

### HierarchicalNode Pydantic model

Reason why it isn't implemented : I went with a data model I found more easy to understand.

### No query answer pair generation 

Explanation : Using metadata and document content, extract query answer pairs for embedding training from all types of supported documents.
Reason why it isn't implemented : Haven't gotten to work on it yet. I want to finish parsing and metadata extraction first. I am also making sure that my system could work ok for RAG so I don't have to redevelop it entierely.


### Sheet classification method via empty cell ratio :

Reason it is not implemented : the block size method I went with is more reliable.

### Embedding of excel data rows :

Reason it is implemented : I want to extend the system to RAG. I don't want it just for query answer pair generation.
Extension : I want to also embed content sheets to be able to retrieve them as well.

### New functionnality HardNegativeProcessor :

Reason it is implemented : I wanted to implement that as a step after query,answer pair generation. Since I have a starting dataset of pairs extracted from other sources, I went with it.

# Extending the spec 

I want to update the spec to show my current objective and decisions.
