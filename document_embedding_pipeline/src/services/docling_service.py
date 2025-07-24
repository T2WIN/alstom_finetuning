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
        
    def convert_doc_to_markdown(self, file_path: Path) -> None:
        """
        Converts a PDF file to a Markdown file, ignoring headers and footers.

        Args:
            pdf_path (Path): The path to the input PDF file.
            markdown_path (Path): The path to save the output Markdown file.

        Raises:
            Exception: If the conversion fails for any reason.
        """
        try:
            logger.info(f"Starting Markdown conversion for: {file_path.name}")
            doc = self.converter.convert(file_path).document
            markdown_content = doc.export_to_markdown()
            return markdown_content

        except Exception as e:
            logger.error(f"Failed to convert '{file_path.name}' to Markdown. Reason: {e}", exc_info=True)
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