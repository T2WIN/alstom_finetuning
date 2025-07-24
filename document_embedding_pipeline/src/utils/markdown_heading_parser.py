import re
from typing import List
from data_models import Section

class MarkdownHeadingParser:
    """
    Static parser for extracting hierarchical sections from Markdown headings.
    Using a class instead of a standalone function allows for:
    1. Future stateful parsing if needed
    2. Consistent namespace organization
    3. Easier extension with helper methods
    """
    @staticmethod
    def parse(content: str) -> List[Section]:
        """
        Parse Markdown content into hierarchical sections based on headings.
        
        Args:
            content: Markdown content string
            
        Returns:
            List of top-level Section objects
        """
        lines = content.split('\n')
        sections = []
        current_h1 = None
        
        for line in lines:
            # Parse H1 headings
            if line.startswith('# '):
                title = line[2:].strip()
                current_h1 = Section(title=title, content_summary="", subsections=[])
                sections.append(current_h1)
                
            # Parse H2 headings
            elif line.startswith('## '):
                title = line[3:].strip()
                if current_h1:
                    current_h1.subsections.append(Section(title=title, content_summary="", subsections=None))
                else:
                    # Handle orphaned H2
                    sections.append(Section(title=title, content_summary="", subsections=None))
        
        return sections