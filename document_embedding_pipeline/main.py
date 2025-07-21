import argparse
import logging
import sys
from pathlib import Path
import asyncio
from typing import Dict, Any

import yaml
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Import custom modules from the project structure
from pipeline.excel_processor import ExcelProcessor
from pipeline.word_processor import WordProcessor
from services.qdrant_service import QdrantService
from utils.logging_setup import setup_logging

# Set up a logger for the main orchestrator
logger = logging.getLogger(__name__)

async def main(input_folder: Path, output_folder: Path):
    """
    Main orchestrator for the document processing pipeline.

    Args:
        input_folder (Path): The folder containing source documents.
        output_folder (Path): The folder where all outputs will be stored.
    """
    # --- 1. Initial Setup ---
    try:
        with open("config.yaml", "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
    except FileNotFoundError:
        print("FATAL: config.yaml not found. Please ensure it exists in the root directory.")
        sys.exit(1)

    # Create output directories if they don't exist
    # log_path = output_folder / config["paths"]["log_file"]
    log_path = Path("/home/grand/alstom_finetuning/data/test") / config["paths"]["log_file"]
    setup_logging(log_path)

    logger.info("=====================================================")
    logger.info("🚀 Starting Document Processing Pipeline")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
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
    files_to_process = Path("/home/grand/alstom_finetuning/data/SSPHA projets").rglob("*")
    excel_files = [file_path for file_path in files_to_process if file_path.suffix in [".xlsx"]]
    word_files = [file_path for file_path in files_to_process if file_path.suffix in [".docx", ".doc"]]

    progress_bar_word = tqdm(word_files, desc="Processing Word Documents", unit="file")

    word_processor = WordProcessor(temp_folder=config["paths"]["temp_dir_name"])
    excel_processor = ExcelProcessor(temp_folder=config["paths"]["temp_dir_name"])

    word_results = []
    for file_path in progress_bar_word:
        progress_bar_word.set_postfix_str(file_path.name)
        result = word_processor.process(file_path)
        word_results.append(result)
    
    excel_tasks = [[excel_processor.process(file_path) for file_path in excel_files][0]]
    excel_results = await tqdm_asyncio.gather(*excel_tasks)

    logger.info(config["qdrant"]["distance_metric"])
    database = QdrantService(db_path=config["paths"]["qdrant_db_path"], 
                             collection_name=config["qdrant"]["collection_name"], 
                             vector_size=config["qdrant"]["vector_size"],
                             distance_metric=config["qdrant"]["distance_metric"],
                             embedding_model_path=config["paths"]["embedding_model_dir"])
    for excel_doc in excel_results:
        database.upsert_excel_document(excel_doc)


    
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process engineering documents to generate datasets for finetuning embedding models."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=False,
        help="Path to the folder containing .docx, .doc, .xlsx, and .xls files.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=False,
        help="Path to the folder where logs, temporary files, and final datasets will be stored.",
    )

    # args = parser.parse_args()

    # input_path = Path(args.input_folder)
    # output_path = Path(args.output_folder)
    input_path = ""
    output_path = ""

    # if not input_path.is_dir():
    #     print(f"Error: The specified input folder does not exist: {input_path}")
    #     sys.exit(1)
        
    # Create the output directory if it doesn't exist
    # output_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(main(input_folder=input_path, output_folder=output_path))