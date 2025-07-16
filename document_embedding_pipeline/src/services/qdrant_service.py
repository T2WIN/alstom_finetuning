# src/services/qdrant_service.py

import logging
from pathlib import Path
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

    def __init__(self, db_path: str, collection_name: str, vector_size: int, distance_metric: str, embedding_model_path: str):
        """
        Initializes the QdrantService.

        Args:
            db_path (str): The path to the Qdrant database file.
            collection_name (str): The name of the collection.
            vector_size (int): The size of the vectors.
            distance_metric (str): The distance metric to use.
            embedding_model_path (str): The path to the embedding model.
        """
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric.upper()
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self._create_collection_if_not_exists()
    
    def _generate_unique_id_for_file(self, payload : Union[WordDocumentPayload, ExcelDocumentPayload]):
        return str(uuid.uuid5(self.uuid_namespace, str(payload.file_path)))
    
    def _create_collection_if_not_exists(self):
        """
        Creates the Qdrant collection if it does not already exist.
        """
        test=self.client.collection_exists(collection_name=self.collection_name)
        if self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists.")
        else:
            logger.info(f"Creating collection: '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )

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
            collection_name=self.collection_name,
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
        """
        Upserts an excel document and its table summaries into the Qdrant collection.

        Args:
            document_payload (Dict[str, Any]): The payload for the document.
            table_summaries (List[str]): A list of table summaries to be embedded.
        """
        table_summaries = [sheet.table_summary for sheet in document_payload.sheets if sheet.sheet_type=="table"]
        if not table_summaries:
            logger.warning("No table summaries provided for upserting.")
            return

        embeddings = self.embedding_model.encode(table_summaries, show_progress_bar=False)

        self.client.upload_points(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self._generate_unique_id_for_file(document_payload),
                    payload=document_payload.model_dump(),
                    vector=embeddings,
                )
            ],
            wait=True,
        )
        logger.info(f"Upserted excel document with {len(table_summaries)} table summaries.")