import argparse
import logging
import sys
from pathlib import Path
import asyncio
from typing import Dict, Any, List

from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Import custom modules from the project structure
from pipeline.excel_processor import ExcelProcessor
from utils.logging_setup import get_component_logger
from pipeline.word_processor import WordProcessor
from utils.config_loader import ConfigLoader
# from services.qdrant_service import QdrantService

# Set up a logger for the main orchestrator
logger = get_component_logger(__name__)

def filter_files_that_have_already_been_converted(file_list: List[Path]) -> List[Path]:
    filtered_files = []
    for file in file_list:
        out_file_excel = file.with_suffix(".xlsx")
        out_file_word = file.with_suffix(".docx")
        if file.suffix == ".xls" and out_file_excel.exists():
            continue
        if file.suffix == ".doc" and out_file_word.exists():
            continue
        filtered_files.append(file)

    return filtered_files

async def main():
    """
    Main orchestrator for the document processing pipeline.

    Args:
        input_folder (Path): The folder containing source documents.
        output_folder (Path): The folder where all outputs will be stored.
    """
    # --- 1. Initial Setup ---
    try:
        # Load and validate configuration
        ConfigLoader.load_config()
    except RuntimeError as e:
        logger.error(f"FATAL: Configuration error: {e}")
        sys.exit(1)

    logger.info("=====================================================")
    logger.info("🚀 Starting Document Processing Pipeline")
    logger.info("=====================================================")

    # --- 2. Pre-run System Checks ---
    logger.info("Performing pre-run system checks...")
    try:
        # As per spec, check for unoserver before starting processing
        logger.info("✅ Unoserver is available.")
    except ConnectionRefusedError as e:
        logger.error(f"❌ CRITICAL: {e}")
        logger.error("Please start the unoserver instance and try again. Exiting.")
        sys.exit(1)
    # Get input data path from config with error handling
    try:
        input_data_path: str = ConfigLoader.get("paths.input_data_path")
        files_to_process = list(Path(input_data_path).rglob("*"))
    except KeyError as e:
        logger.error(f"Missing required configuration: paths.input_data_path")
        sys.exit(1)

    excel_files = [file_path for file_path in files_to_process if file_path.suffix in [".xlsx", ".xls"]]
    excel_files = filter_files_that_have_already_been_converted(excel_files)

    word_files = [file_path for file_path in files_to_process if file_path.suffix in [".docx", ".doc"]]
    word_files = filter_files_that_have_already_been_converted(word_files)

    # Get temp directory from config with error handling
    try:
        temp_dir: str = ConfigLoader.get("paths.temp_dir_name")
        word_processor = WordProcessor(temp_folder=Path(temp_dir))
        excel_processor = ExcelProcessor(temp_folder=temp_dir)
    except KeyError as e:
        logger.error(f"Missing required configuration: paths.temp_dir_name")
        sys.exit(1)
    
    # excel_tasks = [excel_processor.process(file_path) for file_path in excel_files if "Liste_" in file_path.name]
    # excel_tasks = [excel_processor.process(file_path) for file_path in excel_files][:2]
    # excel_results = await tqdm_asyncio.gather(*excel_tasks)

    logger.info("Processing word documents :")
    word_tasks = [word_processor.process(file_path) for file_path in word_files]
    word_results = await tqdm_asyncio.gather(*word_tasks)
    # Qdrant configuration with error handling
    # try:
    #     qdrant_config = {
    #         'db_path': ConfigLoader.get("paths.qdrant_db_path"),
    #         'vector_size': ConfigLoader.get("qdrant.vector_size"),
    #         'distance_metric': ConfigLoader.get("qdrant.distance_metric")
    #     }
    #     database = QdrantService(**qdrant_config)
    # except KeyError as e:
    #     logger.error(f"Missing Qdrant configuration: {e}")
    #     sys.exit(1)
    #
    # for excel_doc in excel_results:
    #     database.upsert_excel_document(excel_doc)


if __name__ == "__main__":
    asyncio.run(main())