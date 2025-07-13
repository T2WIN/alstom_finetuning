# fetch_chunks.py
import json
import random
import sys
from pathlib import Path
from qdrant_client import QdrantClient, models

# --- Configuration ---
# Add the parent directory to the system path to allow imports from config.py
# This assumes the script is run from the project's root directory or the directory containing it.
try:
    import config
except ImportError:
    print("Error: config.py not found. Make sure you are running this script from the project root directory.")
    sys.exit(1)

# --- Constants ---
NUM_CHUNKS_TO_FETCH = 200
RAW_DATA_DIR = Path(config.DATA_DIR)
OUTPUT_FILE = RAW_DATA_DIR / "raw_chunks.jsonl"

def fetch_random_chunks(
    client: QdrantClient,
    collection_name: str,
    num_to_select: int
) -> list[dict]:
    """
    Fetches a random sample of document chunks from a Qdrant collection.

    Args:
        client: An initialized QdrantClient instance.
        collection_name: The name of the collection to fetch from.
        num_to_select: The number of random chunks to select.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        with its ID and text.
    """
    print(f"Connecting to collection '{collection_name}'...")

    # 1. Get all unique chunk IDs efficiently
    all_ids = []
    next_offset = 0
    while next_offset is not None:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Fetch in batches of 1000
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        all_ids.extend([point.id for point in points])
        print(f"\rScrolled through {len(all_ids)} points...", end="")

    print(f"\nFound a total of {len(all_ids)} chunks in the database.")

    if not all_ids:
        print("Error: No chunks found in the collection. Cannot proceed.")
        return []

    # 2. Select a random sample of IDs
    num_to_sample = min(num_to_select, len(all_ids))
    if num_to_sample < num_to_select:
        print(f"Warning: Requested {num_to_select} chunks, but only {len(all_ids)} are available. Fetching {num_to_sample}.")
    
    print(f"Randomly selecting {num_to_sample} chunk IDs...")
    random_ids = random.sample(all_ids, num_to_sample)

    # 3. Retrieve the full payload for the selected IDs
    print("Retrieving full data for selected chunks...")
    retrieved_points = client.retrieve(
        collection_name=collection_name,
        ids=random_ids,
        with_payload=True,
        with_vectors=False,
    )

    # 4. Format the output
    output_data = []
    for point in retrieved_points:
        # Ensure the payload contains the 'text' field
        chunk_text = json.loads(point.payload.get("_node_content", {})).get("text")
        if chunk_text:
            output_data.append({
                "chunk_id": point.id,
                "text": chunk_text
            })
        else:
            print(f"Warning: Chunk with ID {point.id} is missing 'text' in its payload. Skipping.")

    return output_data

def save_chunks_to_jsonl(data: list[dict], file_path: Path):
    """Saves a list of dictionaries to a JSONL file."""
    print(f"Saving {len(data)} chunks to {file_path}...")
    try:
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print("Save complete.")
    except IOError as e:
        print(f"Error saving file to {file_path}: {e}")
        sys.exit(1)

def main():
    """Main function to execute the script."""
    print("--- Starting Random Chunk Retrieval ---")
    
    # Initialize Qdrant client for a local, file-based database
    try:
        client = QdrantClient(path=config.db_path)
    except Exception as e:
        print(f"Error connecting to Qdrant at path '{config.db_path}': {e}")
        sys.exit(1)

    chunks = fetch_random_chunks(
        client=client,
        collection_name=config.collection_name,
        num_to_select=NUM_CHUNKS_TO_FETCH
    )

    if chunks:
        save_chunks_to_jsonl(chunks, OUTPUT_FILE)
    
    print("--- Script finished ---")

if __name__ == "__main__":
    main()
