import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from utils.config_loader import ConfigLoader

# Configure logging
logger = logging.getLogger(__name__)

class DoclingService:
    """
    A service to handle document conversions using the docling library.
    """

    def __init__(self):
        artifacts_path = ConfigLoader.get('paths.artifacts_path')
        pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        pipeline_options = PdfPipelineOptions()
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )    
        
    def convert_pdf_to_markdown(self, pdf_path: Path, markdown_path: Path) -> None:
        """
        Converts a PDF file to a Markdown file, ignoring headers and footers.

        Args:
            pdf_path (Path): The path to the input PDF file.
            markdown_path (Path): The path to save the output Markdown file.

        Raises:
            Exception: If the conversion fails for any reason.
        """
        try:
            logger.info(f"Starting PDF to Markdown conversion for: {pdf_path}")
            doc = self.converter.convert(pdf_path).document
            markdown_content = doc.export_to_markdown()

            # Save the markdown content to the specified file
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            logger.info(f"Successfully converted PDF to Markdown: {markdown_path}")

        except Exception as e:
            logger.error(f"Failed to convert PDF '{pdf_path}' to Markdown. Reason: {e}", exc_info=True)
            # The exception is allowed to propagate up to be handled by the main orchestrator.
            raise