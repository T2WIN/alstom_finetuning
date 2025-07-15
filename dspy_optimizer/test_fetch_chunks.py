# test_fetch_chunks.py
import pytest
import os
import json
import shutil
import uuid
import random  # Added missing import
from pathlib import Path
from qdrant_client import QdrantClient, models

# Import the functions to be tested
# We need to import the module itself to patch the QdrantClient class within it
import fetch_chunks
from fetch_chunks import fetch_random_chunks, main as run_main

# --- Test Constants ---
TEST_COLLECTION_NAME = "test_collection"
NUM_TEST_DOCS = 300
NUM_TO_SELECT_FOR_TEST = 50

@pytest.fixture(scope="module")
def test_environment(tmp_path_factory):
    """
    Pytest fixture to set up a temporary environment for testing.
    
    This fixture creates:
    1. A temporary directory for the test database and data files.
    2. A SINGLE Qdrant database client instance for the entire test module.
    
    It yields the paths and client, and handles cleanup after tests are done.
    """
    # Create a unique base directory for this test run
    base_dir = tmp_path_factory.mktemp("dspy_test_run")
    db_dir = base_dir / "test_db"
    data_dir = base_dir / "test_data"
    os.makedirs(db_dir)
    os.makedirs(data_dir)

    # --- Setup: Create and populate a temporary Qdrant DB ---
    client = QdrantClient(path=str(db_dir))
    client.recreate_collection(
        collection_name=TEST_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
    )

    # Populate with dummy data
    points_to_upload = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=[random.random() for _ in range(4)],
            payload={"_node_content" : '{"text": "This is test chunk number."}'},
        )
        for i in range(NUM_TEST_DOCS)
    ]
    
    client.upload_points(
        collection_name=TEST_COLLECTION_NAME,
        points=points_to_upload,
        wait=True,
    )
    
    print(f"\nSetup: Created temporary DB at {db_dir} with {NUM_TEST_DOCS} documents.")

    # Yield the necessary objects to the tests
    yield {
        "client": client,
        "base_dir": base_dir,
        "data_dir": data_dir,
        "db_path": str(db_dir)
    }

    # --- Teardown: Clean up the temporary directory ---
    print(f"\nTeardown: Removing temporary directory {base_dir}")
    shutil.rmtree(base_dir)


def test_fetch_random_chunks_logic(test_environment):
    """Tests the core chunk fetching logic directly."""
    client = test_environment["client"]
    
    chunks = fetch_random_chunks(
        client=client,
        collection_name=TEST_COLLECTION_NAME,
        num_to_select=NUM_TO_SELECT_FOR_TEST
    )

    assert len(chunks) == NUM_TO_SELECT_FOR_TEST
    
    # Verify the structure and content of a sample chunk
    sample_chunk = chunks[0]
    assert "chunk_id" in sample_chunk
    assert "text" in sample_chunk
    assert isinstance(sample_chunk["chunk_id"], str)
    assert sample_chunk["text"].startswith("This is test chunk number")


def test_main_script_execution(test_environment, monkeypatch):
    """
    Tests the full script execution by running the main() function.
    This test mocks the config values and, crucially, patches the QdrantClient
    instantiation to prevent creating a second client instance.
    """
    # Get the single client instance created by the fixture
    fixture_client = test_environment["client"]

    # Use monkeypatch to override the config values for the duration of this test
    monkeypatch.setattr(fetch_chunks.config, "db_path", test_environment["db_path"])
    monkeypatch.setattr(fetch_chunks.config, "collection_name", TEST_COLLECTION_NAME)
    monkeypatch.setattr(fetch_chunks.config, "DATA_DIR", str(test_environment["data_dir"]))
    
    # Also patch the number of chunks to fetch for consistency
    monkeypatch.setattr(fetch_chunks, "NUM_CHUNKS_TO_FETCH", NUM_TO_SELECT_FOR_TEST)

    # Patch the chunk storage location
    monkeypatch.setattr(fetch_chunks, "OUTPUT_FILE", test_environment["data_dir"] / "raw_chunks.jsonl")
    
    # *** KEY CHANGE ***
    # Patch the QdrantClient class within the fetch_chunks module.
    # When main() calls QdrantClient(), this lambda will intercept it
    # and return our existing fixture_client, avoiding a file conflict.
    monkeypatch.setattr(fetch_chunks, "QdrantClient", lambda path: fixture_client)

    # --- Run the main script ---
    run_main()

    # --- Assertions ---
    output_file = test_environment["data_dir"] / "raw_chunks.jsonl"
    
    # 1. Check if the output file was created
    assert output_file.exists(), "The output file raw_chunks.jsonl was not created."
    
    # 2. Check if the file contains the correct number of lines
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == NUM_TO_SELECT_FOR_TEST, "The output file has an incorrect number of lines."

    # 3. Check the format of the first line
    first_line_data = json.loads(lines[0])
    assert "chunk_id" in first_line_data, "JSON object in file is missing 'chunk_id'."
    assert "text" in first_line_data, "JSON object in file is missing 'chunk_text'."
    assert first_line_data["text"].startswith("This is test chunk number")
