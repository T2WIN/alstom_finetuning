Of course. Here is the updated `README.md` with a new section explaining the error handling system and its design rationale.

-----

# Document Processing Pipeline for Embedding Model Finetuning

This project provides an automated pipeline to process engineering documents (`.docx`, `.doc`, `.xlsx`, `.xls`), extract structural and semantic information using Large Language Models (LLMs), and generate `(query, answer)` pair datasets suitable for finetuning embedding models.

The system is designed to be resilient, featuring a state machine that tracks the progress of each document, allowing it to recover from failures and avoid reprocessing completed files.

-----

## ✨ Key Features

  * **Recursive File Processing:** Automatically discovers and processes all supported documents in a given directory.
  * **Dual Processing Pipelines:** Separate, tailored workflows for Word and Excel documents.
  * **LLM-Powered Extraction:** Uses models like Mistral and Granite to extract titles, structure, and summaries.
  * **Structured Data Output:** Leverages `instructor` and Pydantic for reliable, schema-enforced JSON output from LLMs.
  * **Vector Storage:** Embeds document summaries using a local Qwen model and stores them in a Qdrant vector database.
  * **State Management:** Tracks progress in a `state.json` file and uses a temporary folder for intermediate data, ensuring robustness and recoverability.
  * **Dataset Generation:** Produces multiple `.jsonl` files containing different types of query-answer pairs for model training.

-----

## 🛡️ Error Handling and Resilience

Of course. Here is the updated section for the `README.md` file explaining the error handling structure.

***

## ⚙️ Error Handling and Resilience

The pipeline is designed to be resilient, ensuring that temporary failures do not halt the entire process and that no work is lost. The error handling is built on a few key principles of stateful recovery and clear separation of concerns.

### Key Principles

1.  **Stateful Recovery**: All processing progress is tracked in the `state.json` file. If the script is interrupted, it can resume processing each file from its `last_successful_state`, preventing redundant work.

2.  **Separation of Concerns**: The error handling logic is distinctly layered to make the system predictable and maintainable.
    * **Services (`src/services/`)**: A service's only job is to perform a task (e.g., convert a file). If it cannot complete its task, it simply **raises** a standard Python `Exception`. It does not handle the error.
    * **Processors (`src/pipeline/`)**: A processor orchestrates the steps for a single document. It calls the necessary services. If a service raises an exception, the processor allows the exception to **propagate** upwards. It does not catch it. Its only job is to execute the state machine steps.
    * **Main Orchestrator (`main.py`)**: This is the only place where exceptions are **handled**. The main processing loop wraps each call to a document processor in a `try...except` block. This block is responsible for catching any propagated exceptions, logging the error, and updating the document's state (e.g., incrementing the retry `attempts` count).

### Error Flow Example

Here is the step-by-step flow for a document that fails processing:

1.  **Failure & Raising**: A service fails. For instance, the `unoserver_service` times out while converting a very large file and **raises** an `Exception`.
2.  **Propagation**: The `WordProcessor`, which called the service, does not catch the exception and lets it **propagate** up to `main.py`.
3.  **Handling & Retry**: The `try...except` block in `main.py` **catches** the exception. It logs the error and increments the file's `attempts` count in `state.json`. The application then moves on to the next document.
4.  **Terminal Failure**: If the same file fails again on a subsequent run, its `attempts` count becomes 2. When the script runs again, it will see the file has reached the maximum number of retries. It will then mark the file's status as `failed`, move the original document to the `_failed` folder, and delete any temporary data.

-----

## Prerequisites

Before running this pipeline, ensure the following services are installed and running:

1.  **Python 3.9+**
2.  **Unoserver:** Required for converting Word documents to PDF. It can be run via Docker:
    ```bash
    docker run -d --rm -p 2002:2002 unoserver/unoserver --port 2002
    ```
3.  **Ollama:** For serving the LLMs locally.
      * Install [Ollama](https://ollama.com/).
      * Pull the required models:
        ```bash
        ollama pull mistralai/Mistral-Small-3.2-24B-Instruct-2506
        ollama pull ibm-granite/granite-3.3-8b-instruct
        ```

-----

## 📂 Project Structure

The project is organized to separate different logical components:

```
document-embedding-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── models/
│   └── qwen3-embed-0.6b/
│       └── # Local embedding model files go here
├── src/
│   ├── __init__.py
│   ├── data_models.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── base_processor.py
│   │   ├── word_processor.py
│   │   └── excel_processor.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py
│   │   ├── qdrant_service.py
│   │   └── unoserver_service.py
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py
│       ├── logging_setup.py
│       └── state_manager.py
└── tests/
    ├── __init__.py
    ├── test_utils/
    │   ├── __init__.py
    │   └── test_logging_setup.py
    └── test_pipeline/
        ├── __init__.py
        └── test_pipeline.py
```

-----

## 🚀 Getting Started

### 1\. Installation

Clone the repository and install the required Python packages:

```bash
git clone <your-repository-url>
cd document-embedding-pipeline
pip install -r requirements.txt
```

### 2\. Configuration

Modify the `config.yaml` file to set up your model names, file paths, and other parameters as needed.

### 3\. Usage

Run the main script from the command line, providing the path to the input folder containing your documents and the desired output folder.

```bash
python main.py --input-folder /path/to/your/documents --output-folder ./output
```



## ✍️ Commit Message Guidelines

To maintain a clean and readable version history, this project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. Each commit message should consist of a header, a body, and a footer.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Example

```
feat(pipeline): Add support for Excel file processing

Implements the `ExcelProcessor` class, which handles the parsing,
metadata extraction, and summarization of `.xlsx` and `.xls` files.
This new pipeline classifies sheets as "table" or "content" and
generates structured summaries and schemas.

Closes #42
```

### Type

Must be one of the following:

  * **feat**: A new feature
  * **fix**: A bug fix
  * **docs**: Documentation only changes
  * **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc)
  * **refactor**: A code change that neither fixes a bug nor adds a feature
  * **perf**: A code change that improves performance
  * **test**: Adding missing tests or correcting existing tests
  * **chore**: Changes to the build process or auxiliary tools and libraries

### Scope

The scope should be the name of the package/module affected (e.g., `core`, `logging`, `pipeline`, `word_processor`).

### Subject

The subject contains a succinct description of the change:

  * Use the imperative, present tense: "add" not "added" nor "adds".
  * Don't capitalize the first letter.
  * No dot (.) at the end.