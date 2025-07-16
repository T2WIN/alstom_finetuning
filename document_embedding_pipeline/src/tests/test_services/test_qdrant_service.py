# src/create_and_upsert_documents.py

import logging
from pathlib import Path
from data_models import WordDocumentPayload, ExcelDocumentPayload, Section, ContentSheet, TableSheet, Column
from services.qdrant_service import QdrantService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths and configuration
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "document_collection"
VECTOR_SIZE = 1024  # This should match the embedding model's output dimension
DISTANCE_METRIC = "COSINE"
# In a real application, you'd download a more robust one, e.g., 'all-MiniLM-L6-v2'
EMBEDDING_MODEL_PATH = "/home/grand/alstom_finetuning/document_embedding_pipeline/models/qwen3-embed-0.6b" 

def create_sample_word_document(file_path: Path, title: str, global_summary: str, num_sections: int) -> WordDocumentPayload:
    """Creates a sample WordDocumentPayload."""
    sections = []
    for i in range(num_sections):
        sections.append(Section(
            title=f"Section {i+1} Title for {title}",
            content=f"This is the content of section {i+1} for document '{title}'. It contains some detailed information.",
            summary=f"Summary of section {i+1} for '{title}'.",
            table_summary=None if i % 2 == 0 else f"Table summary for section {i+1} in '{title}'"
        ))
    return WordDocumentPayload(
        file_path=file_path,
        title=title,
        global_summary=global_summary,
        sections=sections
    )

def create_sample_excel_document(file_path: Path, title: str, num_sheets: int) -> ExcelDocumentPayload:
    """Creates a sample ExcelDocumentPayload."""
    sheets = []
    for i in range(num_sheets):
        if i % 2 == 0:  # Create a ContentSheet
            sheets.append(ContentSheet(
                sheet_name=f"ContentSheet{i+1} for {title}",
                content=f"This is the general content for ContentSheet{i+1} in Excel document '{title}'.",
                summary=f"Summary of ContentSheet{i+1} for '{title}'."
            ))
        else:  # Create a TableSheet
            sheets.append(TableSheet(
                sheet_name=f"TableSheet{i+1} for {title}",
                table_schema=[
                    Column(column_name="ID", description="Unique identifier"),
                    Column(column_name="Name", description="Item name"),
                    Column(column_name="Value", description="Numeric value")
                ],
                table_summary=f"This table summarizes data related to various items in TableSheet{i+1} for '{title}'."
            ))
    return ExcelDocumentPayload(
        title=title,
        file_path=file_path,
        sheets=sheets
    )

def main():
    # Ensure the database directory exists
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)

    # Initialize Qdrant Service
    try:
        qdrant_service = QdrantService(
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
            distance_metric=DISTANCE_METRIC,
            embedding_model_path=EMBEDDING_MODEL_PATH
        )
        logger.info("QdrantService initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize QdrantService: {e}")
        return

    # --- Create and Upsert Word Documents ---
    logger.info("\n--- Upserting Word Documents ---")
    word_doc1 = create_sample_word_document(
        file_path=Path("report_q3_2024.docx"),
        title="Quarterly Report Q3 2024",
        global_summary="A comprehensive report detailing the company's performance in Q3 2024.",
        num_sections=3
    )
    qdrant_service.upsert_word_document(word_doc1)

    word_doc2 = create_sample_word_document(
        file_path=Path("project_proposal_v1.docx"),
        title="New Project Proposal Alpha",
        global_summary="Proposal for a new internal project focusing on AI integration.",
        num_sections=2
    )
    qdrant_service.upsert_word_document(word_doc2)

    # --- Create and Upsert Excel Documents ---
    logger.info("\n--- Upserting Excel Documents ---")
    excel_doc1 = create_sample_excel_document(
        file_path=Path("sales_data_2023.xlsx"),
        title="Annual Sales Data 2023",
        num_sheets=4
    )
    qdrant_service.upsert_excel_document(excel_doc1)

    excel_doc2 = create_sample_excel_document(
        file_path=Path("inventory_q1_2025.xlsx"),
        title="Q1 2025 Inventory Tracking",
        num_sheets=3
    )
    qdrant_service.upsert_excel_document(excel_doc2)

    logger.info("\n--- All documents upserted successfully. ---")

    try:
        count_result = qdrant_service.client.count(
            collection_name=COLLECTION_NAME,
            exact=True
        )
        logger.info(f"Total points in collection '{COLLECTION_NAME}': {count_result.count}")
    except Exception as e:
        logger.error(f"Failed to count points: {e}")

if __name__ == "__main__":
    main()