# src/pipeline/word_processor.py

import logging
from pathlib import Path
import yaml
from typing import List, Optional

from services.unoserver_service import convert_document_to_pdf
from services.docling_service import DoclingService
from services.llm_service import LLMService
from pipeline.base_processor import BaseProcessor
from data_models import WordDocumentPayload, Section, HierarchicalNode, WordDocumentStructure
from utils.config_loader import ConfigLoader
import prompts

# Configure logging
logger = logging.getLogger(__name__)

class WordProcessor(BaseProcessor):
    """
    A processor for handling Word documents (.docx, .doc).
    """

    def __init__(self, temp_folder: Path):
        """
        Initializes the WordProcessor.
        
        Args:
            temp_folder (Path): The folder for storing intermediate files.
        """
        self.docling_service = DoclingService()
        self.llm_service = LLMService()
        self.temp_folder = temp_folder
        
        # Load configuration
        self.config = ConfigLoader.get('processing_params.word')
        self.small_doc_threshold = self.config['small_doc_threshold_tokens']
        
        # Model names from config
        self.title_structure_model = ConfigLoader.get('processing_params.word.title_structure_model')
        self.title_only_model = ConfigLoader.get('processing_params.word.title_only_model')
        self.section_summary_model = ConfigLoader.get('processing_params.word.section_summary_model')
        self.table_summary_model = ConfigLoader.get('processing_params.word.table_summary_model')
        self.global_summary_model = ConfigLoader.get('processing_params.word.global_summary_model')

    def _accept_tracked_changes(self):
        """Will be implemented later when I am not pissed about it"""
        pass

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the LLM service tokenizer."""
        return len(self.llm_service.tokenizer.encode(text))

    def _extract_title_and_structure(self, markdown_content: str) -> tuple[str, HierarchicalNode]:
        """Extract title and structure from markdown content."""
        token_count = self._count_tokens(markdown_content)
        
        if token_count <= self.small_doc_threshold:
            # Single LLM call for small documents
            logger.info(prompts.WORD_TITLE_STRUCTURE_EXTRACTION_PROMPT.format(markdown_content=markdown_content))
            response = self.llm_service.get_structured_response(
                prompt=prompts.WORD_TITLE_STRUCTURE_EXTRACTION_PROMPT.format(
                    markdown_content=markdown_content
                ),
                model_name=self.title_structure_model,
                response_model=WordDocumentStructure
            )
            # For small docs, the root node contains the title
            title = response.title
            structure = response
        else:
            # Large document: separate title and structure extraction
            title = self.llm_service.get_structured_response(
                prompt=prompts.WORD_TITLE_ONLY_PROMPT.format(
                    markdown_content=markdown_content[:5000]  # First page content
                ),
                model_name=self.title_only_model,
                response_model=str
            )
            
            # Use Docling for structure extraction
            structure = self.docling_service.extract_structure(markdown_content)
        
        return title, structure

    # def _summarize_tables(self, structure: HierarchicalNode) -> None:
    #     """Recursively summarize all tables in the structure."""
    #     if structure.node_type == "Table":
    #         table_summary = self.llm_service.get_structured_response(
    #             prompt=prompts.WORD_TABLE_SUMMARIZATION_PROMPT.format(
    #                 table_caption=structure.title,
    #                 table_content=""  # TODO: Implement table content extraction
    #             ),
    #             model_name=self.table_summary_model,
    #             response_model=str
    #         )
    #         structure.summary = table_summary
        
    #     for child in structure.children:
    #         self._summarize_tables(child)

    # def _summarize_sections(self, structure: HierarchicalNode) -> None:
    #     """Recursively summarize all sections in the structure."""
    #     if structure.node_type == "Section":
    #         section_summary = self.llm_service.get_structured_response(
    #             prompt=prompts.WORD_SECTION_SUMMARIZATION_PROMPT.format(
    #                 section_title=structure.title,
    #                 section_content=""  # TODO: Implement section content extraction
    #             ),
    #             model_name=self.section_summary_model,
    #             response_model=str
    #         )
    #         structure.summary = section_summary
        
    #     for child in structure.children:
    #         self._summarize_sections(child)

    # def _create_global_summary(self, structure: HierarchicalNode) -> str:
    #     """Create a global summary from section summaries."""
    #     section_summaries = []
        
    #     def collect_summaries(node: HierarchicalNode):
    #         if hasattr(node, 'summary') and node.summary:
    #             section_summaries.append(f"{node.title}: {node.summary}")
    #         for child in node.children:
    #             collect_summaries(child)
        
    #     collect_summaries(structure)
        
    #     return self.llm_service.get_structured_response(
    #         prompt=prompts.WORD_GLOBAL_SUMMARY_PROMPT.format(
    #             document_title=structure.title,
    #             section_summaries="\n".join(section_summaries)
    #         ),
    #         model_name=self.global_summary_model,
    #         response_model=str
    #     )

    # def _get_fallback_title(self, file_path: Path) -> str:
    #     """Use filename as fallback title."""
    #     return file_path.stem.replace('_', ' ')

    # def _structure_to_sections(self, structure: HierarchicalNode) -> List[Section]:
    #     """Convert hierarchical structure to flat list of sections."""
    #     sections = []
        
    #     def traverse(node: HierarchicalNode, level: int = 0):
    #         if node.node_type == "Section":
    #             sections.append(Section(
    #                 title=node.title,
    #                 content="",  # TODO: Implement content extraction
    #                 summary=getattr(node, 'summary', ''),
    #                 table_summary=None
    #             ))
    #         elif node.node_type == "Table":
    #             # Tables are nested within sections
    #             if sections:
    #                 sections[-1].table_summary = getattr(node, 'summary', '')
            
    #         for child in node.children:
    #             traverse(child, level + 1)
        
    #     traverse(structure)
    #     return sections

    def process(self, file_path: Path) -> WordDocumentPayload:
        """
        Executes the processing pipeline for the Word document.
        Returns the processed WordDocumentPayload.
        """
        try:
            logger.info(f"Starting processing for Word document: {file_path.name}")
            
            # Initialize temp file paths
            temp_pdf_path = self.temp_folder / f"{file_path.name}_converted.pdf"
            temp_markdown_path = self.temp_folder / f"{file_path.name}_converted.md"

            # Step 1: Accept tracked changes (if applicable)
            self._accept_tracked_changes()

            # Step 2: Convert Word to PDF
            logger.info(f"Converting Word document to PDF: {file_path.name}")
            convert_document_to_pdf(str(file_path), temp_pdf_path)
            logger.info("Word to PDF conversion successful.")

            # Step 3: Convert PDF to Markdown
            logger.info("Converting PDF to Markdown")
            self.docling_service.convert_pdf_to_markdown(temp_pdf_path, temp_markdown_path)
            
            # Read markdown content
            with open(temp_markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Step 4: Extract title and structure
            logger.info("Extracting title and structure")
            title, structure = self._extract_title_and_structure(markdown_content)
            return title, structure
            # # Fallback to filename if title extraction fails
            # if not title or not title.strip():
            #     title = self._get_fallback_title(file_path)

            # # Step 5: Summarize tables
            # logger.info("Summarizing tables")
            # self._summarize_tables(structure)

            # # Step 6: Summarize sections
            # logger.info("Summarizing sections")
            # self._summarize_sections(structure)

            # # Step 7: Create global summary
            # logger.info("Creating global summary")
            # global_summary = self._create_global_summary(structure)

            # # Step 8: Convert to WordDocumentPayload
            # sections = self._structure_to_sections(structure)
            
            # payload = WordDocumentPayload(
            #     file_path=file_path,
            #     title=title,
            #     global_summary=global_summary,
            #     sections=sections
            # )

            # logger.info(f"Successfully completed processing for: {file_path.name}")
            # return payload

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            raise