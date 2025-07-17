"""
Unit tests for WordProcessor, using the standard unittest library.

Assumes:
  - unoserver is running locally
  - a sample .docx file exists at tests/test_pipeline/fixtures/hello_world.docx
"""

import logging
import tempfile
import shutil
import unittest
from pathlib import Path

from pipeline.word_processor import WordProcessor
from services.unoserver_service import convert_document_to_pdf


def _unoserver_reachable() -> bool:
    """Return True if we can reach unoserver."""
    try:
        convert_document_to_pdf(__file__, "/tmp/__probe.pdf")  # noqa: S108
        return True
    except Exception:
        return False


class TestWordProcessor(unittest.TestCase):
    SAMPLE_FILE = Path(__file__).with_suffix(".docx")  # placeholder
    TEMP_DIR: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.SAMPLE_FILE = Path(__file__).parent / "fixtures" / "hello_world.docx"
        if not cls.SAMPLE_FILE.exists():
            raise unittest.SkipTest(f"Missing fixture {cls.SAMPLE_FILE}")

        if not _unoserver_reachable():
            raise unittest.SkipTest("unoserver not reachable")

    def setUp(self) -> None:
        self.TEMP_DIR = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.TEMP_DIR, ignore_errors=True)

    def test_initial_parsing_happy_path(self):
        """PDF and Markdown files should be produced."""
        wp = WordProcessor(temp_folder=self.TEMP_DIR)

        with self.assertLogs(level="INFO") as log_ctx:
            wp.process(self.SAMPLE_FILE)

        self.assertTrue(wp.temp_pdf_path.exists())
        self.assertGreater(wp.temp_pdf_path.stat().st_size, 0)
        self.assertTrue(wp.temp_markdown_path.exists())
        self.assertGreater(wp.temp_markdown_path.stat().st_size, 0)

        logs = "\n".join(log_ctx.output)
        self.assertIn("Converting Word document to PDF", logs)
        self.assertIn("PDF to Markdown conversion successful", logs)

    def test_exception_propagation(self):
        """Any failure inside process() is propagated upwards."""
        wp = WordProcessor(temp_folder=self.TEMP_DIR)
        with self.assertRaises(Exception):
            wp.process(Path("/does/not/exist.docx"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()