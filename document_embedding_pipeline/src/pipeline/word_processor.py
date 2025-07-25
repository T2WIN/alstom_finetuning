# src/pipeline/word_processor.py

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from services.unoserver_service import convert_document
from services.docling_service import DoclingService
from services.llm_service import LLMService
from pipeline.base_processor import BaseProcessor
from data_models import Section, WordDocumentStructure, Title, WordDocumentPayload, WordDocumentSummary
from utils.config_loader import ConfigLoader
from utils.markdown_heading_parser import MarkdownHeadingParser
import prompts

# Custom exceptions
class LibreOfficeError(Exception):
    """Error related to LibreOffice operations"""
    pass

class MacroExecutionError(Exception):
    """Error executing LibreOffice macro"""
    pass

class ConversionTimeoutError(Exception):
    """Document conversion timed out"""
    pass

class RevisionAcceptanceError(Exception):
    """Failed to accept document revisions"""
    pass

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
        self.title_model = self.config["title_model"]
        self.structure_model = self.config["structure_model"]
        self.summary_model = self.config["summary_model"]

    def _accept_tracked_changes(self, file_path: Path) -> Path:
        """Accept all tracked changes using LibreOffice macro
        
        Args:
            file_path: Path to original document
            
        Returns:
            Path to cleaned document
        """
        try:
            # Create temp output path
            clean_path = self.temp_folder / f"{file_path.stem}.odt"
            
            # Run LibreOffice macro to accept changes
            cmd = [
                "soffice",
                "--headless",
                "--convert-to",
                "odt",
                '--accept="macro:///AcceptChanges.AcceptAllChanges"',
                str(file_path),
                "--outdir",
                str(self.temp_folder)
            ]
            logger.info(cmd)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise LibreOfficeError(f"LibreOffice failed: {result.stderr}")
                
            if not clean_path.exists():
                raise RevisionAcceptanceError("Clean document not created")
                
            return clean_path
                
        except subprocess.TimeoutExpired:
            raise ConversionTimeoutError("LibreOffice timed out accepting changes")
        except Exception as e:
            raise RevisionAcceptanceError(f"Error accepting changes: {str(e)}")

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
            prompt, response_model=Title, model_name=self.title_model
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
  "structure" : {
    ""title": "Introduction to Deep Learning",
    "content_summary": "Covers the basics of deep learning, a subfield of machine learning.",
    "subsections": [
        {
        "title": "1. Neural Networks",
        "content_summary": "At the core of deep learning are neural networks, which are inspired by the human brain. They consist of interconnected layers of nodes.",
        "subsections": null
        }
    ]
}  
}"""
            f"{content}\n"
            "Output in JSON"
        )
        resp: List[Section] = await self.llm.aget_structured_response(
            prompt, response_model=List[Section], model_name=self.structure_model
        )
        return WordDocumentStructure(structure=resp)

    async def _create_global_summary(self, title: str, section_summaries: List[str]) -> WordDocumentSummary:
        """Create a global summary from section summaries"""
        summaries_str = "\n".join(section_summaries)
        prompt = (
            f"Create a comprehensive summary of the entire document based on these section summaries.\n\n"
            f"Document title: {title}\n"
            f"Section summaries:\n{summaries_str}\n\n"
            "Provide a 3-4 sentence summary that captures the main purpose and key points of the document."
        )
        return await self.llm.aget_structured_response(prompt, response_model=WordDocumentSummary, model_name=self.summary_model)

    def _get_all_section_summaries(self, sections: List[Section]) -> List[str]:
        """Flatten the section hierarchy and collect all summaries"""
        summaries = []
        for section in sections:
            summaries.append(section.content_summary)
            if section.subsections:
                summaries.extend(self._get_all_section_summaries(section.subsections))
        return summaries

    async def process(self, file_path: Path):
        """
        Executes the processing pipeline for the Word document.
        Returns the processed WordDocumentPayload.
        """
        try:
            logger.info(f"Starting processing for Word document: {file_path.name}")

            # Step 1: Accept tracked changes and get clean document
            clean_path = self._accept_tracked_changes(file_path)
            logger.info(f"Accepted tracked changes for: {file_path.name}")
            
            mode = self.config.get("conversion_mode", "pdf")
            markdown_content = ""  # Initialize to avoid assignment errors
            if mode == "pdf":
                # Step 2: Convert clean document to PDF
                temp_file = self.temp_folder / clean_path.with_suffix(".pdf")
                logger.info(f"Converting clean document to PDF: {clean_path.name}")
                convert_document(str(clean_path), str(temp_file), "pdf")
                logger.info("Clean document to PDF conversion successful.")

                # Step 3: Convert PDF to Markdown
                logger.info("Converting PDF to Markdown")
                markdown_content = self.docling_service.convert_doc_to_markdown(temp_file)
            elif mode == "docx":
                # For DOCX mode, we already have the clean docx
                temp_file = clean_path
                logger.info(f"Using clean DOCX directly: {clean_path.name}")

                # Step 3: Convert docx to Markdown
                logger.info("Converting docx to Markdown")
                markdown_content = self.docling_service.convert_doc_to_markdown(temp_file)
            
            # Prepend filename to markdown content
            markdown_content = f"File name : {file_path.name}\n" + markdown_content
            
            # Handle small vs large documents
            tokens = self._count_tokens(markdown_content)
            if tokens <= self.small_doc_threshold:
                # Small document: use LLM extraction
                logger.info(f"Processing small document ({tokens} tokens), using LLM extraction")
                structure : WordDocumentStructure = await self._extract_structure(markdown_content)
                title : Title = await self._extract_title(file_path, structure.model_dump_json())
            else:
                # Large document: use Markdown heading parser
                logger.info(f"Processing large document ({tokens} tokens), using Markdown parser")
                sections = MarkdownHeadingParser.parse(markdown_content)
                structure = WordDocumentStructure(structure=sections)
                
                # For title, use first H1 heading or fallback to filename
                title_text = file_path.stem
                if sections and sections[0].title:
                    title_text = sections[0].title
                title = Title(title=title_text)
            logger.info(f"Title : {title}")
            logger.info(f"Structure extracted successfully")

            # Fallback to filename if title extraction fails
            if not title.title:
                title.title = file_path.stem

            # Step 5: Create global summary
            section_summaries = self._get_all_section_summaries(structure.structure)
            global_summary : WordDocumentSummary = await self._create_global_summary(title.title, section_summaries)
            logger.info("Global summary created")

            # Step 6: Create payload
            payload = WordDocumentPayload(
                file_path=str(file_path),
                title=title.title,
                global_summary=global_summary.summary,
                sections=structure.structure
            )
            logger.info(payload.model_dump_json())
            logger.info(f"Successfully completed processing for: {file_path.name}")
            return payload

        except LibreOfficeError as e:
            logger.error(f"LibreOffice error: {str(e)}")
            raise
        except MacroExecutionError as e:
            logger.error(f"Macro execution failed: {str(e)}")
            raise
        except ConversionTimeoutError as e:
            logger.error(f"Conversion timed out: {str(e)}")
            raise
        except RevisionAcceptanceError as e:
            logger.error(f"Revision acceptance failed: {str(e)}")
            # Fallback to original document
            logger.warning("Using original document without change acceptance")
            clean_path = file_path
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path.name}: {e}")
            raise