# src/pipeline/word_processor.py
import logging
from pathlib import Path
from services.unoserver_service import convert_document_to_pdf
from services.docling_service import DoclingService

# Configure logging
logger = logging.getLogger(__name__)

class WordProcessor():
    """
    A processor for handling Word documents (.docx, .doc).
    """

    def __init__(self, temp_folder: Path):
        """
        Initializes the WordProcessor.

        Args:
            file_path (Path): The path to the Word document.
            temp_folder (Path): The folder for storing intermediate files.
            unoserver_service (UnoserverService): The service for file conversions.
        """
        self.docling_service = DoclingService()
        self.temp_folder = temp_folder
        
        

    def _accept_tracked_changes(self):
        """Will be implemented later when I am not pissed about it"""
        pass

    def process(self, file_path: Path) -> None:
        """
        Executes the processing pipeline for the Word document.
        """
        try:
            logger.info(f"Starting processing for Word document: {file_path.name}")
            
            self.temp_pdf_path = self.temp_folder / f"{file_path.name}_converted.pdf"
            self.temp_markdown_path = self.temp_folder / f"{file_path.name}_converted.md"

            # Step 1: Accept tracked changes (if applicable)
            self._accept_tracked_changes()

            # Step 2: Convert Word to PDF
            logger.info(f"Converting Word document to PDF: {file_path.name} -> {self.temp_pdf_path}")
            convert_document_to_pdf(str(file_path), self.temp_pdf_path)
            logger.info("Word to PDF conversion successful.")

            # Step 3: Convert PDF to Markdown
            logger.info(f"Converting PDF to Markdown: {self.temp_pdf_path} -> {self.temp_markdown_path}")
            self.docling_service.convert_pdf_to_markdown(self.temp_pdf_path, self.temp_markdown_path)
            logger.info("PDF to Markdown conversion successful.")

            # --- Future steps for metadata extraction will be added here ---

            logger.info(f"Successfully completed initial parsing for: {file_path.name}")

        except Exception as e:
            logger.error(f"An error occurred during the processing of {file_path.name}. Error: {e}")
            # Propagate the exception to be handled by the main orchestrator
            raise
