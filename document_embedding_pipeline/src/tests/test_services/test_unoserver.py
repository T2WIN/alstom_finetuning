import subprocess
import os

def convert_document(input_path, output_path):
    """Converts a document to a specified format using unoconverter."""
    try:
        subprocess.run(
            ['unoconvert', '--convert-to', 'pdf', input_path, output_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully converted {input_path} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'unoconverter' command not found. Make sure unoserver is installed and in your PATH.")

# Example usage:
if __name__ == "__main__":
    input_file = "/home/grand/alstom_finetuning/document_embedding_pipeline/Final Report for the Big Data Project.docx"
    output_file = "document.pdf"

    convert_document(input_file, output_file)
    # Note: The output file will not be created if the conversion fails.
    # if os.path.exists(output_file):
    #     os.remove(output_file)
