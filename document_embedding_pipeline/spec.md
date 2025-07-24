# **Revised System Specification (v2)**

### 1\. Project Goal

The primary goal is to generate datasets of `(query, answer)` pairs for the purpose of finetuning embedding models. This will be achieved by building an automated pipeline that processes engineering documents, extracts structural and semantic information using LLMs, and formats this information into distinct types of query-answer pairs.


The system will operate as a command-line Python script that **recursively** processes all `.docx`, `.doc`, `.xlsx`, and `.xls` files found within a specified input folder. **Other file types will be ignored, with a log entry generated to record this action.**


The processing for each document follows a state machine, with intermediate data and progress stored to a temporary folder to ensure resilience. Upon successful completion, the temporary data for that document is deleted.


-----


### 2\. Word Document Processing Pipeline


This pipeline handles Microsoft Word (`.docx`, `.doc`) files.


**State Machine:**

`RECEIVED` -\> `CONVERTED_TO_PDF` -\> `CONVERTED_TO_MARKDOWN` -\> `STRUCTURE_EXTRACTED` -\> `SUMMARIZED` -\> `EMBEDDED` -\> `COMPLETE`


#### **2.1. Initial Parsing and Pre-processing**


1.  **Accept Changes:** Programmatically accept all tracked changes in the original `.docx` file. (During implementation, this was found very difficult so it is reported to a future version)

2.  **PDF Conversion:** Convert the finalized `.docx` document into a PDF file using a `unoserver` instance. **The script will check for a running `unoserver` instance on startup and will exit with an error if it is not available.** All subsequent steps will operate on this generated PDF.

3.  **Text Parsing:** Parse the generated PDF using **Docling** to extract its full text content. Docling automatically ignores headers and footers. The output is then converted into a **Markdown representation** to preserve structural elements like tables.


-----


### 2.2. Metadata Extraction


This stage uses LLMs to extract the title, structure, and summaries from the Markdown text. The token count, used to differentiate between small and large documents, will be calculated using the tokenizer for the `mistralai/Mistral-Small-3.2-24B-Instruct-2506` model. Any input to an LLM will be **truncated to a maximum of 10,000 tokens**, and a `WARNING` will be logged if truncation occurs.


#### **2.2.1. Title & Structure Extraction**


  * **For Small Documents (≤ 5k tokens):** A single LLM call will be made to `mistralai/Mistral-Small-3.2-24B-Instruct-2506` to extract both the document title and its hierarchical structure. The LLM will be prompted to parse the Markdown and generate a JSON object conforming to the `HierarchicalNode` Pydantic model. Tables, identifiable by Markdown syntax, will be treated as distinct nodes.

  * **For Large Documents (\> 5k tokens):**

    1.  **Title Extraction:** A dedicated call to `ibm-granite/granite-3.3-8b-instruct` will extract the document title from the first page of the document. **This assumes the parsing library can logically segment the document by page.**

    2.  **Structure Extraction:** The **Docling** library will be used to generate a JSON structure of the document's headings, which will then be mapped to the `HierarchicalNode` model.

  * **Fallback:** If a title cannot be reliably extracted, the system will use the document's filename as the title, after removing the file extension and replacing underscores with spaces.


*(Developer Note: The specific prompts for extraction are to be optimized and defined later.)*


The Pydantic model:


```python

from typing import List, Literal

from pydantic import BaseModel, Field


class HierarchicalNode(BaseModel):

    """A recursive model for deeply nested documents, representing a node in a tree."""

    node_type: Literal["Section", "Table"] = Field(description="The type of the node, either a text section or a table.")

    title: str = Field(description="The title of this section or the caption of the table.")

    children: List['HierarchicalNode'] = Field(default_factory=list, description="A list of child nodes.")

```


#### **2.2.2. Summarization**


1.  **Table Summarization:** Each table node identified during structure extraction is summarized using `ibm-granite/granite-3.3-8b-instruct`. The summary should describe the table's structure and content. This summary is added to the parent section of the table node.

2.  **Section Summarization:** Each text section node is summarized using `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

3.  **Global Summary:** After all sections are summarized, their summaries are concatenated and used to generate a final, global summary of the entire document.


*(Developer Note: The specific prompts for summarization are to be optimized and defined later.)*


-----


### 2.3. Embedding and Storage


The processed data is stored in a Qdrant vector database.


  * **Embedding Model:** `Qwen3 Embedding 0.6b`, loaded locally from the `models/qwen3-embed-0.6b` folder.

  * **Content to Embed:** Only the **textual section summaries** will be converted into vectors.

  * **Vector DB Configuration:** The collection uses vectors of **dimension 1024** with the **Cosine** distance metric. **Qdrant will handle the assignment of unique document IDs.**

  * **Document Payload:** A single document entry is created in Qdrant with a payload containing all its information. Each embedded section summary vector will be linked to this parent document using Qdrant's multi-vector capabilities.

    ```json

    {

      "title": "The Extracted Document Title",

      "global summary" : "This is a summary",

      "section_list": [

        {

          "title": "Section 1 Title",

          "summary": "The summary of section 1.",

          "full_text": "The full text content of section 1..."

        },

        {

          "title": "Section 2 Title (with table)",

          "summary": "The summary of section 2.",

          "table_summary": "Summary of the table within section 2.",

          "full_text": "The full text content of section 2..."

        }

      ]

    }

    ```


-----


### 2.4. Query/Answer Pair Generation


1.  **(Document Title, Global Summary):** Stored in `title_globalsummary_pairs.jsonl`.

2.  **(Section Heading, Section Content):** Stored in `heading_content_pairs.jsonl`.

3.  **(Section Heading, Table Summary):** Stored in `heading_tablesummary_pairs.jsonl`.

    *(Note: This pair is generated for training data augmentation, even though table summaries are not embedded for retrieval in the Word pipeline.)*


-----


### 3\. Excel Processing Pipeline


This pipeline handles Excel (`.xlsx`, `.xls`) files.


**State Machine:**

`RECEIVED` -\> `METADATA_EXTRACTED` -\> `EMBEDDED` -\> `COMPLETE`


#### **3.1. Parsing and Metadata Extraction**


1.  **Sheet Identification:** For each sheet, determine if it contains meaningful content. A sheet is considered empty if **more than 80% of its cells are empty**. This threshold should be included within the config file.

2.  **Title Extraction:** Use `ibm-granite/granite-3.3-8b-instruct` on the text content (first 500 tokens) of all content sheets to extract the document title. Also give the filepath to the llm as additionnal context. 

3.  **Content Classification:** For each meaningful sheet, pass its content (as a CSV-formatted string) to `mistralai/Mistral-Small-3.2-24B-Instruct-2506` to classify it as either a **"table sheet"** or a **"content sheet"**. Or compare the number of commas to the number of words in the csv. Since content sheets have merged cells and specific formatting (that means lots of empty cells), there will be a lot more commas than words.

*(Developer Note: If time is available, will explore finetuning a way smaller model using DSPy for this. [Classification finetuning](https://dspy.ai/tutorials/classification_finetuning/))*

4.  **Content Processing:**

      * **Content Sheets:** Generate a summary of the sheet's text.

      * **Table Sheets:**

          * Remove any columns from the original data where the percentage of `NaN` values exceeds 90%.

          * Provide the table data (as a CSV string, **truncated to the first 100 rows by default**) and the concatenated text from all "content sheets" (as context) to `mistralai/Mistral-Small-3.2-24B-Instruct-2506`. Any input to the LLM will be **truncated to a maximum of 10,000 tokens**, and a `WARNING` will be logged if truncation occurs.

          * The LLM will generate :

            * A **column schema**, where each column has a name and a description.

            * A **table summary** based on the schema and data.

         


#### **3.2. Embedding and Storage**


  * **Embedding Model:** `Qwen3 Embedding 0.6b`.

  * **Content to Embed:** Only the **table summaries** from each "table sheet" will be embedded.

  * **Vector DB Configuration:** Dimension 1024, Cosine distance. **Qdrant handles document IDs.**

  * **Document Payload:** A single document entry is created in Qdrant. **Raw table data is not stored in the payload.**

    ```json

    {

      "title": "The Extracted Document Title",

      "sheet_list": [

        {

          "title": "Sheet name 1 (Content)",

          "summary": "The summary of content sheet 1.",

          "full_text": "The full text content of sheet 1..."

        },

        {

          "title": "Sheet name 2 (Table)",

          "table_schema": {

            "column1": "Description column 1",

            "column2": "Description column 2"

          },

          "table_summary": "Summary of the table within sheet 2."

        }

      ]

    }

    ```


#### **3.3. Query/Answer Pair Generation**


1.  **(Document Title, Sheet Summary):** Stored in `title_summary_pairs.jsonl`. The `answer` is the summary of one sheet, generating one pair per meaningful sheet.

2.  **(Document Title, Table Schema):** Stored in `title_schema_pairs.jsonl`. The `answer` is a paragraph combining all column descriptions. The paragraph is formed by concatenating sentences of the format `"{column_name} means {column_description}."` with spaces.


-----


### 4\. Logging and State Management


  * **Logging:** The system will use extensive logging with standard severity levels (`INFO`, `DEBUG`, `WARNING`, `ERROR`).

    * **Events:** Log processing time, success, or failure for: PDF transformation, parsing, and each metadata extraction sub-task.

    * **Output:** Logs will be printed to the console and saved to a log file. A progress bar will be shown for long-running operations.

  * **State & Failure Handling:**

      * A central `state.json` file is created in the output directory. It contains a list of all files found in the input folder. Each entry tracks the `file_path`, `status` (`pending`, `completed`, `failed`), `last_successful_state`, and `attempts`. This file prevents reprocessing of already completed documents.

      * Intermediate data for each file (e.g., extracted text, summaries) is stored in a **`_temp` sub-folder**, with filenames like `[hash_of_filepath].json`.

      * If any step fails, the system will **retry processing from the last successful state**. After two total failed attempts, the document's status is set to `failed`, its temporary data is deleted, and the original file is moved to a `_failed` folder.


-----


### 5\. Technical Stack


  * **Backend:** Python

  * **Vector Database:** Qdrant, running from a file directly, not a server.

  * **Parsing & Utilities:** Docling (transforming pdfs to markdown), `pandas` (reading excel files), `unoserver` (transforming documents to pdf)

  * **LLMs:** `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, `ibm-granite/granite-3.3-8b-instruct`

  * **Embedding Model:** `Qwen3 Embedding 0.6b`, loaded locally from the `models/qwen3-embed-0.6b` folder.

  * **LLM Serving:** Ollama server run locally.

  * **LLM Serving:** Testing with `unittest`

  * **Network Configuration:** To bypass potential network restrictions, Ollama API calls must be routed through a local proxy using `httpx`, as shown in the example below.

    ```python

    import httpx

    import instructor

    from openai import OpenAI


    # The proxy_url must be set to the Ollama server address

    proxy_url = "http://localhost:11434"

    http_client = httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url})


    client = instructor.patch(

        OpenAI(

            base_url="http://localhost:11434/v1",

            api_key="ollama", # or any other required key for local Ollama

            http_client=http_client

        )

    )

    ```

