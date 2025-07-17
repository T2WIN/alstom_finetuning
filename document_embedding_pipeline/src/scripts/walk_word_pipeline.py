#!/usr/bin/env python3
"""
Demonstrate WordProcessor step-by-step on a real file.

Usage:
    python scripts/walk_word_pipeline.py --file docs/hello.docx --temp /tmp/demo
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from pipeline.word_processor import WordProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk through the Word pipeline")
    parser.add_argument("--file", required=True, type=Path, help="Path to .docx file")
    parser.add_argument("--temp", type=Path, default=Path("/tmp/word_demo"))
    args = parser.parse_args()

    temp_dir: Path = args.temp.resolve()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    processor = WordProcessor(temp_folder=temp_dir)

    print(f"=== Processing {args.file} ===")
    try:
        title, structure = processor.process(args.file)
    except Exception as exc:
        logging.error("Pipeline aborted: %s", exc)
        return

    print("\n=== Results ===")
    print(title)
    print(structure)

if __name__ == "__main__":
    main()