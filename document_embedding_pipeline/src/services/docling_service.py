# Update src/services/docling_service.py

import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from utils.config_loader import ConfigLoader
from data_models import HierarchicalNode

# Configure logging
logger = logging.getLogger(__name__)

class DoclingService:
    """
    A service to handle document conversions using the docling library.
    """

    def __init__(self):
        # Load artifacts path from config
        artifacts_path = ConfigLoader.get('paths.artifacts_path')
        pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
    # Rest of DoclingService remains the same 
        
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
            logger.error(f"Failed to convert PDF '{pdf_path}' to Markdown. Reason: {e}")
            # The exception is allowed to propagate up to be handled by the main orchestrator.
            raise
    
    def extract_structure(self, markdown_content: str) -> HierarchicalNode:
        """
        Extract hierarchical structure from markdown content.
        This is a basic implementation - in practice you'd want more sophisticated parsing.
        """
        # Basic markdown structure extraction
        lines = markdown_content.split('\n')
        root = HierarchicalNode(
            node_type="Section",
            title="Document Root",
            children=[]
        )
        
        current_section = None
        current_table = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # This is a heading
                level = line.count('#', 0, line.find(' '))
                title = line.strip('# ').strip()
                
                section = HierarchicalNode(
                    node_type="Section",
                    title=title,
                    children=[]
                )
                
                if level == 1:
                    root.children.append(section)
                else:
                    # Find parent at level-1
                    parent = root
                    # Simplified - in practice you'd track the hierarchy properly
                    parent.children.append(section)
                    
            elif line.startswith('|'):
                # This could be a table
                if current_table is None:
                    current_table = HierarchicalNode(
                        node_type="Table",
                        title="Table",  # TODO: Extract table caption
                        children=[]
                    )
                    root.children.append(current_table)
        
        return root