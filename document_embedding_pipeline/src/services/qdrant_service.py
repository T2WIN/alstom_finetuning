# src/services/qdrant_service.py

import logging
from typing import Union
from data_models import WordDocumentPayload, ExcelDocumentPayload
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid

logger = logging.getLogger(__name__)


class QdrantService:
    """
    A service to manage interactions with a Qdrant vector database.
    """
    uuid_namespace = uuid.UUID("5e964fc6-ed86-4d74-b010-67ae37583070")

    def __init__(self, db_path: str, vector_size: int, distance_metric: str, embedding_model_path: str):
        """
        Initializes the QdrantService. Loads the embedding model in memory.

        Args:
            db_path (str): The path to the Qdrant database file.
            vector_size (int): The size of the vectors.
            distance_metric (str): The distance metric to use.
            embedding_model_path (str): The path to the embedding model folder.
        """
        self.client = QdrantClient(path=db_path)
        self.docs_collection_name = "documents"
        self.rows_collection_name = "document_rows"
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self._create_collections_if_not_exist()
    
    def _generate_unique_id_for_file(self, payload : Union[WordDocumentPayload, ExcelDocumentPayload]):
        """Generates a unique and deterministic uuid for each file"""
        return str(uuid.uuid5(self.uuid_namespace, str(payload.file_path)))
    
    def _create_collections_if_not_exist(self):
        """Creates both documents and rows collections if they do not exist already."""
        if not self.client.collection_exists(collection_name=self.docs_collection_name):
            self.client.create_collection(
                collection_name=self.docs_collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric,
                    multivector_config=models.MultiVectorConfig( #Allows us to store multiple vector representations per document (example : one vector per table summary)
                        comparator=models.MultiVectorComparator.MAX_SIM 
                    ),
                ),
            )
            logger.info(f"Created collection: '{self.docs_collection_name}'")
        
        # The row collection will store a vector per table row (tables are those extracted from excel documents)
        if not self.client.collection_exists(collection_name=self.rows_collection_name):
            self.client.create_collection(
                collection_name=self.rows_collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, 
                    distance=self.distance_metric
                ),
            )
            logger.info(f"Created collection: '{self.rows_collection_name}'")

    def upsert_word_document(self, document_payload: WordDocumentPayload):
        """
        Upserts a document and its section summaries into the Qdrant collection.

        Args:
            document_payload (Dict[str, Any]): The payload for the document.
            section_summaries (List[str]): A list of section summaries to be embedded.
        """
        section_summaries = [section.summary for section in document_payload.sections]
        embeddings = self.embedding_model.encode(section_summaries, show_progress_bar=False)

        self.client.upload_points(
            collection_name="documents",
            points=[
                models.PointStruct(
                    id=self._generate_unique_id_for_file(document_payload),
                    payload=document_payload.model_dump(),
                    vector=embeddings,
                )
            ],
            wait=True,
        )
        logger.info(f"Upserted document with {len(section_summaries)} section summaries.")

    def upsert_excel_document(self, document_payload: ExcelDocumentPayload):
        # === 1. Upsert the Document ===
        table_summaries = [sheet.table_summary for sheet in document_payload.sheets if sheet.sheet_type == "table"]
        if not table_summaries:
            logger.warning("No table summaries provided for upserting.")
            return

        doc_embeddings = self.embedding_model.encode(table_summaries, show_progress_bar=False)
        doc_id = self._generate_unique_id_for_file(document_payload)

        self.client.upload_points(
            collection_name=self.docs_collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    payload=document_payload.model_dump(exclude={"sheets": {"__all__": {"serialized_rows"}}}),
                    vector=doc_embeddings,
                )
            ],
            wait=True,
        )
        logger.info(f"Upserted document '{doc_id}' with {len(table_summaries)} table summaries.")

        # === 2. Upsert Individual Rows from tables ===
        row_points = []
        for sheet in document_payload.sheets:
            if sheet.sheet_type == "table" and hasattr(sheet, 'serialized_rows'):
                if not sheet.serialized_rows:
                    continue
                
                # Embed all rows in the sheet at once
                row_embeddings = self.embedding_model.encode(sheet.serialized_rows, show_progress_bar=True)
                
                logger.info(f"{len(sheet.serialized_rows)}")
                logger.info(f"{sheet.serialized_rows[0:10]}")
                for i, row_text in enumerate(sheet.serialized_rows):
                    # logger.info(f"Row points amount : {len(row_points)}")
                    row_points.append(
                        models.PointStruct(
                            # Create a unique, deterministic ID for each row
                            id=str(uuid.uuid5(self.uuid_namespace, f"{doc_id}_{sheet.sheet_name}_row_{i}")),
                            vector=row_embeddings[i],
                            payload={
                                "doc_id": doc_id,
                                "file_path": str(document_payload.file_path),
                                "sheet_name": sheet.sheet_name,
                                "row_content": row_text,
                            }
                        )
                    )

        if row_points:
            self.client.upload_points(
                collection_name=self.rows_collection_name,
                points=row_points,
                wait=True,
            )
            logger.info(f"Upserted {len(row_points)} rows for document '{doc_id}'.")