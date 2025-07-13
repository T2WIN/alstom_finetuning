# data_processor.py
import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.node_parser.slide import SlideNodeParser
from llama_index.readers.docling import DoclingReader

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from vector_store import get_storage_context, get_node_count, get_index, get_vector_store
import config
from data_filtering import EduFilter
from pathlib import Path

logger = logging.getLogger(__name__)



class DataProcessor:
    """
    Handles the discovery, processing, and ingestion of new documents
    into the persistent vector store.
    """

    def __init__(self):
        """Initializes the data processor and its components."""
        logger.info("Initializing DataProcessor...")
        self._setup_settings()
        
        # Use our helper function to get the index
        self.vector_store = get_vector_store()
        self.storage_context = get_storage_context()
        self.index = get_index()
        self.file_extractor = self._init_file_extractor()
        self.ingestion_pipeline = self._init_ingestion_pipeline()

        node_count = get_node_count()
        self.is_store_empty = (node_count == 0)

        if self.is_store_empty:
            logger.info("Vector store is currently empty. All found files will be processed as new.")
        else:
            logger.info(f"Vector store contains {node_count} nodes. Checking for new files.")
        
        logger.info("DataProcessor initialized successfully.")

    def _setup_settings(self):
        """Configures global LlamaIndex settings."""
        Settings.llm = OpenRouter(
            model=config.LLM_FOR_PARSING,
            api_key=config.OPEN_ROUTER_API_KEY,
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL, 
            trust_remote_code=True, 
            embed_batch_size=1
        )

    def _init_file_extractor(self) -> dict:
        """Initializes the DoclingReader for PDF extraction."""
        # pipeline_options = PdfPipelineOptions(generate_picture_images=False)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption()}
        )
        docling_reader = DoclingReader(doc_converter=converter, export_type="json")
        logger.info("Initialized file extractor with DoclingReader.")
        return {".pdf": docling_reader}

    def _init_ingestion_pipeline(self) -> IngestionPipeline:
        """Initializes the document processing and embedding pipeline."""
        node_parser = SlideNodeParser.from_defaults(
            chunk_size=200, window_size=5, llm=Settings.llm, llm_workers=3
        )
        pipeline = IngestionPipeline(
            transformations=[
                DoclingNodeParser(chunker=HybridChunker(tokenizer=config.EMBEDDING_MODEL, max_tokens = 256)),
                EduFilter(),
                # node_parser,
                Settings.embed_model,
            ], 
            vector_store=self.vector_store
        )
        logger.info("Initialized ingestion pipeline.")
        return pipeline

    def _is_file_processed(self, file_path: str) -> bool:
        """
        Checks if a file has already been processed by querying vector store metadata.
        """
        # If we already determined the store is empty, we know the file isn't processed.
        if self.is_store_empty:
            return False

        # If the store is NOT empty, then we must perform the query to check for this specific file.
        filters = MetadataFilters(filters=[ExactMatchFilter(key="file_path", value=file_path)])
        retriever = self.index.as_retriever(similarity_top_k=1, filters=filters)
        
        nodes = retriever.retrieve("check")
        return len(nodes) > 0

    def find_data_files(self) -> list[str]:
        """Finds all files in the data directory."""
        files = [str(f) for f in Path(config.DATA_DIR).rglob("*") if f.is_file()]
        logger.info(f"Found {len(files)} data files in {config.DATA_DIR}.")
        return files

    def process_new_documents(self):
        """
        Main method to find and process all new documents.
        """
        logger.info("Starting document processing run...")
        file_paths = self.find_data_files()
        
        for file_path in file_paths:
            if self._is_file_processed(file_path):
                logger.info(f"Skipping '{file_path}' as it is already in the vector store.")
                continue

            try:
                logger.info(f"Processing new file: '{file_path}'")
                directory_reader = SimpleDirectoryReader(
                    input_files=[file_path], file_extractor=self.file_extractor
                )
                documents = directory_reader.load_data()

                logger.info(f"Running ingestion pipeline for {len(documents)} document sections...")
                self.ingestion_pipeline.run(documents=documents, show_progress=True)
                
                if self.is_store_empty:
                    self.is_store_empty = False
                    logger.info("First document processed. Vector store is no longer considered empty for this run.")

                logger.info(f"Successfully processed and stored nodes for '{file_path}'.")

            except Exception as e:
                logger.error(f"Failed to process file {file_path}. Error: {e}", exc_info=True)
                
        logger.info("Document processing run finished.")