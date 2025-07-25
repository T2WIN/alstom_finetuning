# Update src/services/llm_service.py to use config values

import logging
from typing import Type, TypeVar
import httpx
import instructor
from openai import OpenAI, OpenAIError, AsyncOpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils.config_loader import ConfigLoader
from utils.logging_setup import get_component_logger

# Configure logging
logger = logging.getLogger(__name__)

# Update src/services/llm_service.py to use config values with error handling

import logging
from typing import Type, TypeVar
import httpx
import instructor
from openai import OpenAI, OpenAIError, AsyncOpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils.config_loader import ConfigLoader
from utils.logging_setup import get_component_logger

# Configure logging
logger = get_component_logger(__name__)

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
        Initializes the LLMService with configuration from config.yaml.
        
        Raises:
            RuntimeError: If required configuration sections are missing
        """
        try:
            # Load required configuration with error handling
            self.ollama_base_url: str = ConfigLoader.get('llm.ollama_base_url')
            self.tokenizer_model: str = ConfigLoader.get('llm.tokenizer_model')
            self.max_tokens: int = ConfigLoader.get('llm.max_input_tokens')
            
            # As specified in spec.md, all requests must be routed through a
            # local proxy. httpx.Client is used for this purpose.
            http_client = httpx.AsyncClient(proxy=self.ollama_base_url)

            self.client: OpenAI = instructor.patch(
                OpenAI(
                    base_url=f"{self.ollama_base_url}/v1",
                    api_key="ollama",
                ),
                mode=instructor.Mode.JSON,
            )
            self.async_client: AsyncOpenAI = instructor.patch(
                AsyncOpenAI(
                    base_url=f"{self.ollama_base_url}/v1",
                    api_key="ollama",
                    http_client=http_client,
                ),
                mode=instructor.Mode.JSON,
            )

            # Load the tokenizer specified for calculating prompt sizes
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
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
            The text, truncated to max_tokens if it was too long.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            logger.warning(
                f"Input text with {len(tokens)} tokens exceeds the "
                f"{self.max_tokens} token limit. It will be truncated."
            )
            truncated_tokens = tokens[:self.max_tokens]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return text

    async def aget_structured_response(self,
        prompt: str,
        response_model: Type[PydanticModel],
        model_name: str = "qwen3:4b",
        **kwargs,
    ) -> PydanticModel:
        try:
            logger.info(f"Requesting structured response {response_model} using model '{model_name}'.")
            truncated_prompt = self._truncate_text(prompt)
            logger.debug(f"Sending truncated prompt: {truncated_prompt[:100]}...")

            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": truncated_prompt}],
                response_model=response_model,
                **kwargs,
            )
            logger.debug("Successfully received structured response")
            return response
        except OpenAIError as e:
            logger.error(f"API call to model '{model_name}' failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in aget_structured_response: {e}", exc_info=True)
            raise


