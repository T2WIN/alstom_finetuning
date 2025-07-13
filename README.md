This project creates a highly advanced data pipeline designed to transform complex technical PDF documents into a rich, queryable knowledge base.

Its primary goal is **not just to store information, but to augment it**. For every piece of information extracted from the documents, the system synthetically generates a high-quality, relevant question from the perspective of a specific professional persona (like a "Rolling Stock Design Engineer" or "Maintenance Technician").

The final output is a sophisticated dataset where each text chunk is paired with a contextual query, making it exceptionally valuable for training advanced Q&A models, creating robust evaluation benchmarks, or powering a highly intelligent RAG system.

***

### How the Code Achieves This

The project operates in two main stages, managed by distinct, refactored components:

#### 1. Structured Ingestion and Storage

This stage is handled by the **`DataProcessor`**. Its job is to build the foundational knowledge base with maximum context and efficiency.

* **Intelligent Parsing:** Instead of basic text extraction, the system uses the `docling` library to perform a deep analysis of each PDF. The `DocumentConverter` understands the layout, tables, and text flows, outputting a detailed **JSON structure** for each document.
* **Two-Step Chunking:** An `IngestionPipeline` processes this JSON.
    1.  The `DoclingNodeParser` first creates intelligent, structurally-aware text chunks (nodes).
    2.  A `SlideNodeParser` then further refines these nodes using sentence-windowing, ensuring each chunk retains its immediate surrounding context.
* **Efficient Storage:** The final, contextualized nodes are converted into numerical vectors (embeddings) and stored in a **LanceDB vector store**. This high-performance, local database ensures that the system is fast and scalable without needing to load the entire index into memory. The processor is incremental, using metadata checks to only process new or updated files.



***

#### 2. Persona-Driven Question Generation

This is the core augmentation stage, orchestrated by the **`QuestionGenerator`**. It systematically works through every node in the vector store and enriches it with a generated question.

* **Persona Matching:** For each text chunk, the system uses an embedding model to find the most relevant professional persona from a predefined list. This ensures the question's context is appropriate (e.g., an engineer asks about design specs, while a technician asks about maintenance procedures).
* **Two-Step LLM Chain:** To ensure high-quality questions, the system uses a sophisticated two-step API call process:
    1.  **Analyze:** The first LLM call analyzes the text chunk from the persona's viewpoint to determine the best *type* (e.g., Fact Retrieval, Reasoning) and *format* of question to ask.
    2.  **Generate:** The second LLM call uses the original text, the persona, and the analysis from the first step to craft the final, polished question.
* **Robust API Management:** This process involves thousands of API calls. The **`LLMDispatcher`** is a critical component that manages a pool of different LLM clients. It automatically handles rate limits (per-minute and daily), concurrency, and persistent usage logging, allowing the system to run efficiently and reliably without being throttled.
* **Data Augmentation:** The generated question is then saved back into the metadata of its corresponding text chunk in LanceDB, creating a permanent, valuable link between a piece of knowledge and a relevant query.

#### Current issues

1) Saving to LanceDB doesn't work yet. It is causing :
```bash
Traceback (most recent call last):
  File "/home/grand/alstom_finetuning/main.py", line 47, in main
    processor.process_new_documents()
  File "/home/grand/alstom_finetuning/data_processor.py", line 120, in process_new_documents
    if self._is_file_processed(file_path):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/data_processor.py", line 103, in _is_file_processed
    nodes = retriever.retrieve("check")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index_instrumentation/dispatcher.py", line 319, in wrapper
    result = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index/core/base/base_retriever.py", line 246, in retrieve
    nodes = self._retrieve(query_bundle)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index_instrumentation/dispatcher.py", line 319, in wrapper
    result = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index/core/indices/vector_store/retrievers/retriever.py", line 104, in _retrieve
    return self._get_nodes_with_embeddings(query_bundle)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index/core/indices/vector_store/retrievers/retriever.py", line 181, in _get_nodes_with_embeddings
    query_result = self._vector_store.query(query, **self._kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index/vector_stores/lancedb/base.py", line 482, in query
    where = _to_lance_filter(query.filters, self._metadata_keys)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/grand/alstom_finetuning/.venv/lib/python3.12/site-packages/llama_index/vector_stores/lancedb/base.py", line 52, in _to_lance_filter
    if filter.key in metadata_keys:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: argument of type 'NoneType' is not iterable
```