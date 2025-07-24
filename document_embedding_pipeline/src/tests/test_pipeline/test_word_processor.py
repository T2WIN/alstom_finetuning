import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from pipeline.word_processor import WordProcessor
from services.unoserver_service import convert_document
from utils.markdown_heading_parser import MarkdownHeadingParser

@pytest.fixture
def sample_file():
    return Path(__file__).parent / "fixtures" / "hello_world.docx"

def test_unoserver_reachable():
    try:
        convert_document(__file__, "/tmp/__probe.pdf", "pdf")
    except Exception:
        pytest.skip("unoserver not reachable")

@pytest.mark.asyncio
async def test_initial_parsing_happy_path(sample_file, tmp_path):
    """PDF files should be produced."""
    wp = WordProcessor(temp_folder=tmp_path)
    await wp.process(sample_file)
    
    # Verify PDF file was created
    assert (tmp_path / "hello_world.pdf").exists()

@pytest.mark.asyncio
@patch('pipeline.word_processor.LLMService', autospec=True)
@patch('pipeline.word_processor.DoclingService', autospec=True)
async def test_small_document_processing(mock_docling, mock_llm, sample_file, tmp_path):
    """Test processing flow for small documents."""
    # Setup mocks
    mock_docling.return_value.convert_doc_to_markdown.return_value = "# Title\nContent"
    mock_llm.return_value.aget_structured_response = AsyncMock(side_effect=[
        MagicMock(structure=[MagicMock()]),
        MagicMock(title="Title")
    ])
    
    wp = WordProcessor(temp_folder=tmp_path)
    wp.small_doc_threshold = 5000  # Set high threshold to force small doc path
    
    await wp.process(sample_file)
    
    # Verify LLM was called for structure extraction
    mock_llm.return_value.aget_structured_response.assert_called()

@pytest.mark.asyncio
@patch.object(MarkdownHeadingParser, 'parse')
@patch('pipeline.word_processor.DoclingService', autospec=True)
async def test_large_document_processing(mock_docling, mock_parser, sample_file, tmp_path):
    """Test processing flow for large documents."""
    # Setup mocks
    mock_docling.return_value.convert_doc_to_markdown.return_value = "# Title\n## Section\nContent"
    mock_parser.return_value = [MagicMock(title="Title", content_summary="", subsections=[])]
    
    wp = WordProcessor(temp_folder=tmp_path)
    wp.small_doc_threshold = 100  # Set low threshold to force large doc path
    
    await wp.process(sample_file)
    
    # Verify Markdown parser was called
    mock_parser.assert_called()

@pytest.mark.asyncio
async def test_exception_propagation(tmp_path):
    """Any failure inside process() is propagated upwards."""
    wp = WordProcessor(temp_folder=tmp_path)
    with pytest.raises(Exception):
        await wp.process(Path("/does/not/exist.docx"))