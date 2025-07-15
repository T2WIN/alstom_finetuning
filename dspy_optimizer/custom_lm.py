# custom_lm.py
import dspy
import asyncio
import logging
from typing import List

from llm_services import LLMDispatcher, LLMClientConfig

logger = logging.getLogger(__name__)

class DispatcherLM(dspy.LM):
    def __init__(self, dispatcher: LLMDispatcher, model: str = "dispatcher_managed"):
        """💡 FIX: The __init__ method no longer needs a 'loop' parameter."""
        super().__init__(model=model)
        self.dispatcher = dispatcher
        self.provider = "custom_dispatcher"
        self.history = []

    def __call__(self, prompt: str = None, **kwargs) -> List[str]:
        """
        💡 FIX: Simplify the main synchronous entry point.
        We rely on nest_asyncio to correctly run the async code from this
        synchronous context (which may be in a separate thread).
        """
        final_prompt = prompt
        if final_prompt is None:
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list) and messages:
                final_prompt = messages[-1].get("content")

        if final_prompt is None:
            raise TypeError("DispatcherLM was called without a 'prompt' or 'messages' keyword argument.")

        # This is now safe because nest_asyncio patches asyncio.run.
        return asyncio.run(self.arequest(final_prompt, **kwargs))

    async def arequest(self, prompt: str, **kwargs) -> List[str]:
        """The core async logic remains the same."""
        # This will trigger the _initialize() methods in your dispatcher
        # on the one, true event loop managed by pytest-asyncio.
        client_config: LLMClientConfig = await self.dispatcher.get_available_client()

        if not client_config:
            logger.error("DispatcherLM: Failed to acquire an available client.")
            raise RuntimeError("No available LLM clients found.")
        
        semaphore = self.dispatcher.client_semaphores[client_config.name]
        
        async with semaphore:
            try:
                if client_config.limiter:
                    await client_config.limiter.acquire()

                messages = [{"role": "user", "content": prompt}]
                
                api_kwargs = {
                    "model": client_config.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1500),
                    # ... other params
                }
                
                # Make sure the client's own async components are ready
                await client_config.initialize_async_components()
                
                response = await client_config.raw_client.chat.completions.create(**api_kwargs)
                
                await self.dispatcher.release_client_on_success(client_config)
                
                completions = [choice.message.content for choice in response.choices]
                self.history.append({'prompt': prompt, 'response': completions, 'kwargs': api_kwargs})
                
                return completions

            except Exception as e:
                logger.error(f"DispatcherLM: API call failed for client '{client_config.name}'. Error: {e}", exc_info=True)
                if client_config:
                    await self.dispatcher.release_client_on_failure(client_config)
                raise e

    def get_history(self, last_n=None):
        return self.history[-last_n:] if last_n is not None else self.history