# src/pipeline/word_processor.py

import logging
from pathlib import Path
import yaml
from typing import List, Optional

from services.unoserver_service import convert_document
from services.docling_service import DoclingService
from services.llm_service import LLMService
from pipeline.base_processor import BaseProcessor
from data_models import Section, WordDocumentStructure, Title
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
        self.llm = LLMService()
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
        return len(self.llm.tokenizer.encode(text))

    async def _extract_title(
        self,
        file_path : Path,
        content : str
    ) -> Title:
        """Extract the document title from the first two pages"""

        prompt = (
            "Here is the structure of a Word document\n"
            "Extract verbatime its title."
            "Make sure it representative of its content."
            "Sometimes the file name is a good name and sometimes not."
            f"File name : {file_path.name}"
            f"{content}\n"
            "Output in JSON"
        )
        resp: Title = await self.llm.aget_structured_response(
            prompt, response_model=Title
        )
        return resp

    async def _extract_structure(
        self,
        content : str
    ) -> WordDocumentStructure:
        """Extract the document title from the first two pages"""

        prompt = (
            f"Extract the hierarchical structure from this document:\n"
            "**Do not nest the sections deeper than 2 levels.**"
            "The main document is level 0, its direct subsections are level 1, and their subsections are level 2. Do not create any level 3 sections."
            """# Example:
Here is an example of how to process a document.

## Example Input Document:
Introduction to Deep Learning

This document covers the basics of deep learning, a subfield of machine learning.

1. Neural Networks
At the core of deep learning are neural networks, which are inspired by the human brain. They consist of interconnected layers of nodes.

## Example JSON Output:
```json
{
  "title": "Introduction to Deep Learning",
  "content": "This document covers the basics of deep learning, a subfield of machine learning.",
  "subsections": [
    {
      "title": "1. Neural Networks",
      "content": "At the core of deep learning are neural networks, which are inspired by the human brain. They consist of interconnected layers of nodes.",
      "subsections": null
    }
  ]
}"""
            f"{content}\n"
            "Output in JSON"
        )
        resp: List[Section] = await self.llm.aget_structured_response(
            prompt, response_model=List[Section]
        )
        return WordDocumentStructure(structure=resp)

    async def process(self, file_path: Path):
        """
        Executes the processing pipeline for the Word document.
        Returns the processed WordDocumentPayload.
        """
        try:
            logger.info(f"Starting processing for Word document: {file_path.name}")

            # Step 1: Accept tracked changes (if applicable)
            self._accept_tracked_changes()
            
            mode = "pdf" # or docx
            if mode == "pdf":
                # Step 2: Convert Word to PDF
                temp_file = self.temp_folder / file_path.with_suffix(".pdf")
                logger.info(f"Converting Word document to PDF: {file_path.name}")
                convert_document(str(file_path), str(temp_file), "pdf")
                logger.info("Word to PDF conversion successful.")

                # Step 3: Convert PDF to Markdown
                logger.info("Converting PDF to Markdown")
                markdown_content = self.docling_service.convert_doc_to_markdown(temp_file)
            elif mode == "docx":
                temp_file = self.temp_folder / file_path.with_suffix(".docx")
                logger.info(f"Converting Word document to docx: {file_path.name}")
                convert_document(str(file_path), str(temp_file), "docx")
                logger.info(".doc to .docx conversion successful.")

                # Step 3: Convert PDF to Markdown
                logger.info("Converting docx to Markdown")
                markdown_content = self.docling_service.convert_doc_to_markdown(temp_file)
            
            markdown_content = f"File name : {file_path.name}\n" + markdown_content

            # Step 4: Extract title and structure !!!!!!!!!!!!!! WILL MOVE IT AFTER THE STRUCTURE EXTRACT TO HAVE BETTER CONTEXT.
            
            structure : WordDocumentStructure = await self._extract_structure(markdown_content[:int(len(markdown_content)*0.2)])
            title : Title = await self._extract_title(file_path, structure.model_dump_json())
            logger.info(f"Title : {title}")
            logger.info(f"structure : {structure.model_dump_json()}")
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