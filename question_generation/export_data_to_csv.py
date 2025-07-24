import csv
import logging
import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

# Assuming your project structure allows this import
# This is the same module used by your QuestionGenerator
import vector_store
import config

# --- Logging Setup ---
# Configure logging to show informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Settings.llm = OpenRouter(
            model=config.LLM_FOR_PARSING,
            api_key=config.OPEN_ROUTER_API_KEY,
        )
Settings.embed_model = HuggingFaceEmbedding(
    model_name=config.EMBEDDING_MODEL, 
    trust_remote_code=True, 
    embed_batch_size=1
)

# --- Configuration ---
# Define the name for the output CSV file
OUTPUT_FILE = "nodes_export.csv"

def export_nodes_to_csv():
    """
    Connects to the vector store, retrieves all nodes, and writes
    their text, file name, and generated query to a CSV file.
    """
    logger.info("Starting the node export process...")

    try:
        # 1. Initialize and get a connection to the vector store index
        logger.info("Connecting to the vector store...")
        index = vector_store.get_index()
        vs = index.vector_store
        logger.info("Successfully connected to the vector store.")

        # 2. Retrieve all nodes from the vector store
        # We call get_nodes without any filters to fetch everything.
        logger.info("Retrieving all nodes. This might take a moment...")
        all_nodes = vs.get_nodes()
        
        if not all_nodes:
            logger.warning("No nodes were found in the vector store. The CSV file will be empty.")
        else:
            logger.info(f"Found {len(all_nodes)} nodes to export.")

        # 3. Write the data to a CSV file
        logger.info(f"Writing nodes to '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            # Define the CSV column headers
            # These match the requested attributes
            fieldnames = ['text', 'file_name', 'generated_query']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row to the CSV
            writer.writeheader()

            # 4. Iterate through each node and write its data to a row
            for node in all_nodes:
                writer.writerow({
                    'text': node.get_content(),
                    # Use .get() to safely access metadata keys that might be missing
                    'file_name': node.metadata.get('file_name', 'N/A'),
                    'generated_query': node.metadata.get('generated_queries', 'N/A')
                })
        
        # Get the full path for the final message
        output_path = os.path.abspath(OUTPUT_FILE)
        logger.info(f"Export complete! All nodes have been saved to: {output_path}")

    except ImportError:
        logger.error("Failed to import the 'vector_store' module.")
        logger.error("Please ensure this script is run from the root of your project directory.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    # Run the main export function when the script is executed
    export_nodes_to_csv()
