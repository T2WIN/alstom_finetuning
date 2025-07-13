import logging
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

import config

logger = logging.getLogger(__name__)

# This constant is now used to configure the Qdrant collection.
EMBEDDING_DIMENSION = 1024

# Initialize the Qdrant client once.
# This client stores data on disk at the specified path, similar to Chroma's PersistentClient.
db_path = str(config.STORAGE_DIR / "qdrant_db")
client = qdrant_client.QdrantClient(path=db_path)
collection_name = "documents"

# Check if the collection already exists. If not, create it.
try:
    client.get_collection(collection_name=collection_name)
    logger.info(f"Qdrant collection '{collection_name}' already exists.")
except Exception:
    logger.info(f"Creating Qdrant collection: '{collection_name}'")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    )

def get_node_count() -> int:
    """
    Gets the total number of vectors in the Qdrant collection.
    This function is fast as it reads collection metadata.
    """
    try:
        # The count method returns a CountResult object.
        count_result = client.count(collection_name=collection_name, exact=True)
        return count_result.count
    except Exception as e:
        logger.error(f"Could not get node count from Qdrant: {e}", exc_info=True)
        # Return 0 to be safe in case of a connection or other error.
        return 0

def get_vector_store() -> QdrantVectorStore:
    """
    Initializes and returns the Qdrant vector store.
    """
    logger.debug(f"Connecting to existing Qdrant collection: '{collection_name}'.")
    return QdrantVectorStore(client=client, collection_name=collection_name)

def get_storage_context() -> StorageContext:
    """Returns a LlamaIndex StorageContext configured with our vector store."""
    # This function requires no changes.
    return StorageContext.from_defaults(vector_store=get_vector_store())

def get_index() -> VectorStoreIndex:
    """Returns a LlamaIndex VectorStoreIndex connected to our vector store."""
    # This function requires no changes.
    return VectorStoreIndex.from_vector_store(vector_store=get_vector_store())