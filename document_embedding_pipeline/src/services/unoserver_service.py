from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)

def convert_document(input_path: Path, output_path: Path, file_type: str):
    """Converts a document to a specified format using unoconverter."""
    try:
        subprocess.run(
            ['unoconvert', '--convert-to', file_type, str(input_path), str(output_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting {input_path} : {e}", exc_info=True)
        raise
    except FileNotFoundError:
        logger.error("Error: 'unoconverter' command not found. Make sure unoserver is installed and in your PATH.")
        raise