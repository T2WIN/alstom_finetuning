# question_generator.py
import asyncio
import logging
import time # NEW: Import the time module
from typing import List, Type

import numpy as np
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm as anext_tqdm
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

# Import from our refactored modules
import config
import prompts
import vector_store
from llm_services import LLMDispatcher, LLMClientConfig, get_llm_configs
from schema import Query, CharacterQuestionAnalysis

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates questions for documents in the vector store that haven't
    been processed yet, using a pool of LLM clients.
    """

    def __init__(self):
        """Initializes the question generator."""
        logger.info("Initializing QuestionGenerator...")
        client_configs = get_llm_configs()
        self.dispatcher = LLMDispatcher(client_configs, log_file=config.LOG_FILE)
        
        self.index = vector_store.get_index()
        self.vector_store = self.index.vector_store

        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        self._init_characters()
        logger.info("QuestionGenerator initialized successfully.")

    def _init_characters(self):
        """Loads character descriptions and pre-computes their embeddings."""
        self.characters = [
            prompts.INDUSTRIALIZATION_ENGINEER_DESCRIPTION,
            prompts.MAINTENANCE_LOGISTICS_TECH_DESCRIPTION,
            prompts.ROLLING_STOCK_DESIGN_ENGINEER_DESCRIPTION,
            prompts.MAINTENANCE_DEPOT_TECHNICIAN,
            prompts.TRAIN_SAFETY_ENGINEER_DESCRIPTION,
        ]
        prompt_embeddings = self.model.encode(self.characters, show_progress_bar=False)
        self.prompt_embeddings_norm = prompt_embeddings / np.linalg.norm(
            prompt_embeddings, axis=1, keepdims=True
        )

    async def structured_output_llm_async(self, prompt: str, schema: Type[BaseModel], client_config: LLMClientConfig):
        """Async call to an LLM for structured JSON output, with rate limiting."""
        async with self.dispatcher.client_semaphores[client_config.name]:
            if client_config.limiter:
                await client_config.limiter.acquire()
            
            return await client_config.client.chat.completions.create(
                model=client_config.model,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
            )

    def choose_appropriate_character(self, query: str) -> str:
        """Finds the most similar character persona for a given text chunk."""
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        similarities = np.dot(self.prompt_embeddings_norm, query_embedding_norm.T).flatten()
        most_similar_prompt_index = np.argmax(similarities)
        return self.characters[most_similar_prompt_index]

    def save_query_to_vector_store(self, query: Query, node: BaseNode):
        """Saves the generated query to the document's metadata in the docstore."""
        node.metadata["generated_query"] = query.query
        self.vector_store.delete_nodes([node.node_id])
        self.vector_store.add([node])
        logger.debug(f"Saved generated query for document {node.node_id}.")

    async def _character_producer(self, nodes: List[BaseNode], queue: asyncio.Queue):
        """Fetches docs, runs character selection, and queues data for LLM processing."""
        logger.info(f"Character producer started for {len(nodes)} documents.")
        loop = asyncio.get_running_loop()

        for node in nodes:
            try:
                doc_text = node.get_content()
                
                # --- METRICS: Measure character selection time ---
                start_time = time.time()
                relevant_character = await loop.run_in_executor(
                    None, self.choose_appropriate_character, doc_text
                )
                duration = time.time() - start_time
                # logger.info(f"METRIC: Character selection for doc {node.node_id} took {duration:.4f}s")
                # --- END METRICS ---
                
                await queue.put({
                    "doc_id": node.node_id,
                    "document": node,
                    "doc_text": doc_text,
                    "character": relevant_character,
                })
            except Exception as e:
                logger.error(f"Error in producer for doc {node.node_id}: {e}", exc_info=True)
                
        logger.info("Character producer has finished queuing all documents.")

    async def _llm_consumer(self, queue: asyncio.Queue, pbar: anext_tqdm):
        """Consumes data from queue, calls LLMs, and saves results."""
        while True:
            client_config = None
            data = await queue.get()
            if data is None:
                queue.task_done()
                break

            # --- METRICS: Full consumer task timer ---
            task_start_time = time.time()
            
            try:
                doc_id, document, doc_text, relevant_character = (
                    data["doc_id"], data["document"], data["doc_text"], data["character"]
                )

                # --- METRICS: Measure client acquisition time ---
                wait_start_time = time.time()
                client_config = await self.dispatcher.get_available_client()
                while not client_config:
                    # logger.warning(f"No available clients. Consumer for doc {doc_id} is waiting 10s.")
                    await asyncio.sleep(10)
                    client_config = await self.dispatcher.get_available_client()
                wait_duration = time.time() - wait_start_time
                logger.info(f"METRIC: Client acquisition for doc {doc_id} took {wait_duration:.4f}s")
                # --- END METRICS ---
                
                # --- LLM Call 1 ---
                llm1_start_time = time.time()
                config_gen_context = prompts.CONFIG_GEN_PROMPT.format(character=relevant_character, passage=doc_text)
                query_gen_config: CharacterQuestionAnalysis = await self.structured_output_llm_async(config_gen_context, CharacterQuestionAnalysis, client_config)
                llm1_duration = time.time() - llm1_start_time
                logger.info(f"METRIC: LLM call 1 (Analysis) for doc {doc_id} took {llm1_duration:.4f}s using {client_config.name}")
                
                # --- LLM Call 2 ---
                llm2_start_time = time.time()
                question_gen_prompt = prompts.QUESTION_GEN_PROMPT.format(
                    character=relevant_character, passage=doc_text,
                    type=query_gen_config.query_type, difficulty=query_gen_config.query_format,
                )
                generated_query: Query = await self.structured_output_llm_async(question_gen_prompt, Query, client_config)
                llm2_duration = time.time() - llm2_start_time
                logger.info(f"METRIC: LLM call 2 (Question) for doc {doc_id} took {llm2_duration:.4f}s using {client_config.name}")
                
                # --- Vector Store Save ---
                save_start_time = time.time()
                self.save_query_to_vector_store(generated_query, document)
                save_duration = time.time() - save_start_time
                logger.info(f"METRIC: Vector store save for doc {doc_id} took {save_duration:.4f}s")
                
                # --- Success Case ---
                logger.info(f"Successfully generated query for doc {doc_id} using {client_config.name}.")
                await self.dispatcher.release_client_on_success(client_config)

            except Exception as e:
                logger.error(f"Error processing doc {data.get('doc_id', 'unknown')} with client {client_config.name if client_config else 'N/A'}: {e}", exc_info=True)
                if client_config:
                    await self.dispatcher.release_client_on_failure(client_config)
            finally:
                queue.task_done()
                pbar.update(1)
                # --- METRICS: Log total time for one item ---
                task_duration = time.time() - task_start_time
                logger.info(f"METRIC: Full consumer task for doc {data.get('doc_id', 'unknown')} took {task_duration:.4f}s")
                # --- END METRICS ---

    async def generate(self):
        """Main method to orchestrate question generation."""
        logger.info("Starting question generation run...")
        
        filters = MetadataFilters(
            filters=[MetadataFilter(key="generated_query", operator=FilterOperator.IS_EMPTY, value=None)]
        )
        nodes_to_process = self.index.vector_store.get_nodes(filters=filters)

        if not nodes_to_process:
            logger.info("No new documents to process.")
            return
        
        total_to_process = len(nodes_to_process)
        logger.info(f"Found {total_to_process} documents requiring question generation.")
        
        total_concurrency = sum(c.concurrency for c in self.dispatcher.clients)
        num_consumers = max(1, total_concurrency)
        processing_queue = asyncio.Queue(maxsize=num_consumers * 4)

        with anext_tqdm(total=total_to_process, desc="Generating Questions") as pbar:
            consumers = [asyncio.create_task(self._llm_consumer(processing_queue, pbar)) for _ in range(num_consumers)]
            
            producer_task = asyncio.create_task(self._character_producer(nodes_to_process, processing_queue))
            
            await producer_task
            await processing_queue.join()
            
            for consumer in consumers:
                consumer.cancel()
            await asyncio.gather(*consumers, return_exceptions=True)

        logger.info(f"Question generation run complete. Processed {total_to_process} documents.")