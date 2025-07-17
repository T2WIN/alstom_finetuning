# src/prompts.py
# NEED TO DEFINE WHAT A SECTION IS BETTER OTHERWISE IT WON'T WORK.
WORD_TITLE_STRUCTURE_EXTRACTION_PROMPT = """You are an expert document analyst. Analyze the following markdown document and extract:
1. The document title
2. The hierarchical structure of the document

For the structure, identify:
- Sections with their titles
- Tables with their captions
- The nesting relationship between elements

Return a JSON object with:
- title: The main document title
- structure: A recursive tree of nodes (sections and tables)

Each node should have:
- node_type: either "Section" or "Table", nodes can not have any other type.
- title: the section heading or table caption
- children: nested nodes (if any)

Document content:
{markdown_content}
"""

WORD_TITLE_ONLY_PROMPT = """Extract the title from this document. Look at the first page or main heading. Return only the title as a string.

Document content:
{markdown_content}
"""

WORD_SECTION_SUMMARIZATION_PROMPT = """Summarize the following document section in 2-3 sentences. Focus on the key information and main points.

Section title: {section_title}
Section content:
{section_content}

Return a concise summary.
"""

WORD_TABLE_SUMMARIZATION_PROMPT = """Analyze this table and provide a brief summary of its structure and content. Include:
- What the table represents
- Key columns/data points
- Main insights or trends

Table caption: {table_caption}
Table content:
{table_content}

Return a concise summary.
"""

WORD_GLOBAL_SUMMARY_PROMPT = """Create a comprehensive summary of the entire document based on these section summaries. 

Document title: {document_title}
Section summaries:
{section_summaries}

Provide a 3-4 sentence summary that captures the main purpose and key points of the document.
"""