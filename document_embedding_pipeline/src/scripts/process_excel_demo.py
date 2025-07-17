#!/usr/bin/env python3
"""
Standalone smoke-test for the new ExcelProcessor.
Logs every step and writes a small summary per file so you can verify quickly.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from pipeline.excel_processor import ExcelProcessor
from services.llm_service import LLMService  # adjust import if needed
from utils.logging_setup import setup_logging   # adjust import if needed

setup_logging("logs", "INFO")
logger = logging.getLogger(__name__)

async def process_one(file_path: Path) -> None:
    logger.info("🟢 START  %s", file_path.name)
    try:
        llm = LLMService()
        proc = ExcelProcessor(llm_service=llm)
        result = await proc.process(file_path)

        # quick sanity checks
        sheet_cnt = len(result["sheets"])
        total_rows = sum(
            len(tbl["rows"]) for sh in result["sheets"] for tbl in sh["tables"]
        )
        logger.info(
            "✅ FINISH %s  (%d sheets, %d total rows processed)",
            file_path.name,
            sheet_cnt,
            total_rows,
        )

        # write summary JSON next to the file for human inspection
        summary_path = file_path.with_suffix(".summary.json")
        summary_path.write_text(
            result["text_content"][:500] + ("..." if len(result["text_content"]) > 500 else ""),
            encoding="utf-8",
        )
        logger.debug("Wrote preview → %s", summary_path)

    except Exception as exc:
        logger.exception("❌ FAILED %s  %s", file_path.name, exc)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Async Excel pipeline smoke test")
    parser.add_argument("--input-folder", required=True, type=Path)
    parser.add_argument("--output-folder", type=Path, default=Path("./verify_out"))
    args = parser.parse_args()

    in_dir: Path = args.input_folder.resolve()
    out_dir: Path = args.output_folder.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    logger.info("Scanning %s for Excel files...", in_dir)
    excel_files = list(in_dir.rglob("*.xlsx"))[:2]
    if not excel_files:
        logger.warning("No Excel files found. Exiting.")
        return

    logger.info("Found %d file(s). Processing concurrently...", len(excel_files))
    tasks = [process_one(f) for f in excel_files]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)