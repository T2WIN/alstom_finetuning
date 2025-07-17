# Update src/services/llm_service.py to use config values

import logging
from typing import Type, TypeVar
import httpx
import instructor
from openai import OpenAI, OpenAIError, AsyncOpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils.config_loader import ConfigLoader

# Configure logging
logger = logging.getLogger(__name__)

# Load configuration
OLLAMA_BASE_URL = ConfigLoader.get('llm.ollama_base_url')
TOKENIZER_MODEL = ConfigLoader.get('llm.tokenizer_model')
MAX_TOKENS = ConfigLoader.get('llm.max_input_tokens')

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)

# Rest of LLMService remains the same, but now uses config values

# --- Setup ---
logger = logging.getLogger(__name__)

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class LLMService:
    """
    A service layer for interacting with local Large Language Models (LLMs)
    served via Ollama.

    This class handles the complexities of connecting to the LLM, formatting
    requests, and ensuring that responses conform to a specified Pydantic
    schema using the `instructor` library. It also enforces input token limits
    to prevent excessively long prompts.
    """

    def __init__(self):
        """
        Initializes the LLMService.

        Sets up the HTTP client with the required proxy for Ollama, patches the
        OpenAI client with `instructor` for structured output, and loads the
        tokenizer for calculating input length.
        """
        try:
            # As specified in spec.md, all requests must be routed through a
            # local proxy. httpx.Client is used for this purpose.
            # Note this doesn't work with llama server !!!!!!!!!!!
            http_client = httpx.AsyncClient(
                proxy=OLLAMA_BASE_URL
            )

            # The OpenAI client is configured to point to the local Ollama v1 API
            # endpoint. `instructor.patch` enhances this client to handle
            # Pydantic response models automatically.
            self.client: OpenAI = instructor.patch(
                OpenAI(
                    base_url=f"{OLLAMA_BASE_URL}/v1",
                    api_key="ollama",  # Required by the API, but can be any string for Ollama
                    # http_client=http_client,
                ),
                mode=instructor.Mode.JSON,
            )
            self.async_client: AsyncOpenAI = instructor.patch(
                AsyncOpenAI(
                    base_url=f"{OLLAMA_BASE_URL}/v1",
                    api_key="ollama",  # Required by the API, but can be any string for Ollama
                    http_client=http_client,
                ),
                mode=instructor.Mode.JSON,
            )

            # Load the tokenizer specified for calculating prompt sizes.
            # This ensures that truncation logic is consistent with the intended model.
            self.tokenizer : PreTrainedTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
            logger.info("LLMService initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {e}", exc_info=True)
            # As per the error handling design in README.md, the service raises
            # the exception to be handled by the main orchestrator.
            raise

    def _truncate_text(self, text: str) -> str:
        """
        Truncates the input text to a maximum token limit if necessary.

        Args:
            text: The input string to process.

        Returns:
            The text, truncated to MAX_TOKENS if it was too long.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) > MAX_TOKENS:
            logger.warning(
                f"Input text with {len(tokens)} tokens exceeds the "
                f"{MAX_TOKENS} token limit. It will be truncated."
            )
            # Truncate the token list and decode it back into a string.
            truncated_tokens = tokens[:MAX_TOKENS]
            return self.tokenizer.decode(
                truncated_tokens, skip_special_tokens=True
            )
        return text

    def get_structured_response(
        self,
        prompt: str,
        model_name: str,
        response_model: Type[PydanticModel],
    ) -> PydanticModel:
        """
        Sends a prompt to the specified LLM and returns a structured response
        validated against a Pydantic model.

        This method first truncates the prompt to stay within the token limit,
        then calls the LLM. The `instructor` library handles the validation,
        ensuring the LLM's JSON output matches the `response_model` schema.

        Args:
            prompt: The user-provided prompt for the LLM.
            model_name: The name of the Ollama model to use (e.g.,
                        'mistralai/Mistral-Small-3.2-24B-Instruct-2506').
            response_model: The Pydantic class that defines the desired
                            structure of the LLM's response.

        Returns:
            An instance of the `response_model` populated with data from the
            LLM's response.

        Raises:
            OpenAIError: If the API call to Ollama fails for any reason
                         (e.g., connection error, model not found).
            Exception: For any other unexpected errors during processing.
        """
        logger.debug(f"Requesting structured response using model '{model_name}'.")
        
        
        # Ensure the prompt does not exceed the maximum token limit.
        truncated_prompt = self._truncate_text(prompt)
        logger.info(f"Sending following prompt :{truncated_prompt}")

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": truncated_prompt,
                    }
                ],
                response_model=response_model,
            )
            logger.debug(f"Successfully received and validated structured response.")
            return response
        except OpenAIError as e:
            logger.error(
                f"API call to model '{model_name}' failed: {e}", exc_info=True
            )
            # Propagate the error to the caller (main.py) for state handling.
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while getting a structured response: {e}",
                exc_info=True,
            )
            raise

    async def aget_structured_response(self,
        prompt: str,
        response_model: Type[PydanticModel],
        model_name: str = "qwen3:4b",
        **kwargs,
    ) -> PydanticModel:
        logger.debug(f"Requesting structured response using model '{model_name}'.")
        
        
        # Ensure the prompt does not exceed the maximum token limit.
        truncated_prompt = self._truncate_text(prompt)
        logger.info(f"Sending following prompt :{truncated_prompt}")

        try:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": truncated_prompt}],
                response_model=response_model,   # or however your wrapper spells it
                **kwargs,
            )
            logger.debug(f"Successfully received and validated structured response.")
            return response
        except OpenAIError as e:
            logger.error(
                f"API call to model '{model_name}' failed: {e}", exc_info=True
            )
            # Propagate the error to the caller (main.py) for state handling.
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while getting a structured response: {e}",
                exc_info=True,
            )
            raise
