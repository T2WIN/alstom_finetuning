import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from utils.config_loader import ConfigLoader
from utils.logging_setup import get_component_logger

# Configure logging
logger = get_component_logger(__name__)

class DoclingService:
    """
    A service to handle document conversions using the docling library.
    """

    def __init__(self):
        """
        Initializes DoclingService with configuration from config.yaml.
        
        Raises:
            RuntimeError: If required configuration is missing
        """
        try:
            # Load configuration with type annotations
            artifacts_path: str = ConfigLoader.get('paths.artifacts_path')
            
            # Initialize pipeline options with artifacts path
            pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("DoclingService initialized successfully")
        except KeyError as e:
            logger.error(f"Missing required configuration: {e}")
            raise RuntimeError(f"Configuration error: {e}") from e
        except Exception as e:
            logger.error(f"Failed to initialize DoclingService: {e}")
            raise
        
    def convert_doc_to_markdown(self, file_path: Path) -> str:
        """
        Converts a document to Markdown format.

        Args:
            file_path: Path to input document file

        Returns:
            Markdown content as string

        Raises:
            Exception: If conversion fails
        """
        try:
            logger.info(f"Starting Markdown conversion for: {file_path.name}")
            doc = self.converter.convert(file_path).document
            return doc.export_to_markdown()
        except Exception as e:
            logger.error(f"Failed to convert '{file_path.name}' to Markdown: {e}", exc_info=True)
            raise