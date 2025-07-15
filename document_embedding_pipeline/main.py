import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from tqdm import tqdm

# Import custom modules from the project structure
from src.pipeline.excel_processor import ExcelProcessor
from src.pipeline.word_processor import WordProcessor
from src.services.unoserver_service import check_unoserver_availability
from src.utils.file_handler import get_supported_files
from src.utils.logging_setup import setup_logging
from src.utils.state_manager import StateManager

# Set up a logger for the main orchestrator
logger = logging.getLogger(__name__)

def main(input_folder: Path, output_folder: Path):
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
    log_path = output_folder / config["paths"]["log_file"]
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
        check_unoserver_availability(config['services']['unoserver'])
        logger.info("✅ Unoserver is available.")
    except ConnectionRefusedError as e:
        logger.error(f"❌ CRITICAL: {e}")
        logger.error("Please start the unoserver instance and try again. Exiting.")
        sys.exit(1)

    # --- 3. State Initialization ---
    state_manager = StateManager(output_folder)
    initial_files = get_supported_files(input_folder)
    state_manager.initialize_state(initial_files)
    
    files_to_process = state_manager.get_pending_files()
    if not files_to_process:
        logger.info("No new or pending files to process. All documents are up-to-date.")
        logger.info("Pipeline finished.")
        return

    logger.info(f"Found {len(files_to_process)} documents to process.")

    # --- 4. Main Processing Loop ---
    # The loop iterates through all files marked for processing.
    # The error handling logic from the README is implemented here:
    # exceptions are caught only at this top level.
    progress_bar = tqdm(files_to_process, desc="Processing Documents", unit="file")
    for file_path in progress_bar:
        progress_bar.set_postfix_str(file_path.name)
        
        try:
            # Determine which processor to use based on the file extension
            extension = file_path.suffix.lower()
            if extension in [".docx", ".doc"]:
                processor = WordProcessor(file_path, output_folder, config, state_manager)
            elif extension in [".xlsx", ".xls"]:
                processor = ExcelProcessor(file_path, output_folder, config, state_manager)
            else:
                # This case should technically not be reached due to get_supported_files
                logger.warning(f"Unsupported file type for {file_path}. Skipping.")
                continue

            # This is the main processing call. It executes the state machine for the doc.
            processor.process()
            
            # If process() completes without error, the processor itself updates the state to 'COMPLETE'.
            logger.info(f"✅ Successfully completed processing for: {file_path.name}")

        except Exception as e:
            # This is the single catch-all block as per the error handling design.
            # It handles any exception that propagates up from services or processors.
            logger.error(f"🚨 An error occurred while processing {file_path.name}: {e}", exc_info=True)
            state_manager.record_failure(file_path)

    logger.info("=====================================================")
    logger.info("Pipeline run finished.")
    state_manager.log_summary()
    logger.info("=====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process engineering documents to generate datasets for finetuning embedding models."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to the folder containing .docx, .doc, .xlsx, and .xls files.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the folder where logs, temporary files, and final datasets will be stored.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_folder)
    output_path = Path(args.output_folder)

    if not input_path.is_dir():
        print(f"Error: The specified input folder does not exist: {input_path}")
        sys.exit(1)
        
    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    main(input_folder=input_path, output_folder=output_path)