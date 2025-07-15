# main.py
import asyncio
import logging
from dotenv import load_dotenv

# Import the refactored components of our application
import config
from data_processor import DataProcessor
from question_generator import QuestionGenerator

# Load environment variables from a .env file at the start
load_dotenv()

def setup_logging():
    """Configures the logging for the entire application."""
    # This setup directs logs to both a file and the console (stream).
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.BASE_DIR / "app.log"),
            logging.StreamHandler()
        ]
    )
    # You can set different log levels for noisy libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

async def main():
    """
    The main asynchronous entry point for the document processing and
    question generation pipeline.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=============================================")
    logger.info("🚀 STARTING RAG PIPELINE EXECUTION")
    logger.info("=============================================")

    try:
        # --- STAGE 1: DATA INGESTION ---
        # The DataProcessor finds new PDF files, chunks them, embeds them,
        # and stores them in our persistent vector store (e.g., LanceDB).
        logger.info("[STAGE 1/2] Initializing Data Processor to check for new documents...")
        processor = DataProcessor()
        processor.process_new_documents()
        logger.info("[STAGE 1/2] Data processing complete. Vector store is up to date.")

        # --- STAGE 2: QUESTION GENERATION ---
        # The QuestionGenerator finds all document chunks that do not yet
        # have a generated question and creates one using the LLM dispatcher.
        logger.info("[STAGE 2/2] Initializing Question Generator...")
        q_gen = QuestionGenerator()
        await q_gen.generate()
        logger.info("[STAGE 2/2] Question generation complete.")

    except Exception as e:
        logger.critical(f"A critical error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        logger.info("=============================================")
        logger.info("✅ PIPELINE EXECUTION FINISHED")
        logger.info("=============================================")

if __name__ == "__main__":
    # This is the standard way to run the main async function.
    asyncio.run(main())