import logging
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

# Import the function to get the index from your vector_store script
from vector_store import get_index
import config

# --- Basic Configuration ---
# It's good practice to have a basic logger
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

Settings.llm = OpenRouter(
            model=config.LLM_FOR_PARSING,
            api_key=config.OPEN_ROUTER_API_KEY,
        )
Settings.embed_model = HuggingFaceEmbedding(
    model_name=config.EMBEDDING_MODEL, 
    trust_remote_code=True, 
    embed_batch_size=1
)

def main():
    """
    Loads the index, creates a query engine, and retrieves an answer.
    """
    print("Connecting to the index...")
    # 1. Load the index from your vector store
    index = get_index()

    # 2. Create a query engine from the index
    # The query engine is the main interface for asking questions
    query_engine = index.as_query_engine()

    # 3. Define your query
    query_text = "What are the main electronic components of an electric train traction ?"

    print(f"\nExecuting query: {query_text}")

    # 4. Execute the query and get the response
    response = query_engine.query(query_text)

    # 5. Print the response
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()