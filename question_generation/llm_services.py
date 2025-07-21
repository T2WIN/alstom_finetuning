# llm_services.py
import asyncio
import json
import logging
import datetime
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
import instructor
import time

# Import centralized configuration
import config

# --- Logging Setup ---
logger = logging.getLogger(__name__)

@dataclass
class LLMClientConfig:
    # ... (no changes needed in this class)
    name: str
    model: str
    base_url: str
    api_key: str
    rpm_limit: Optional[int] = None
    daily_limit: Optional[int] = None
    concurrency: int = 1
    limiter: Optional[AsyncLimiter] = field(init=False, default=None)
    daily_counter: int = field(init=False, default=0)
    client: Optional[instructor.AsyncInstructor] = field(init=False, default=None)
    is_disabled: bool = field(init=False, default=False)
    is_in_timeout: bool = field(init=False, default=False)
    timeout_until: float = field(init=False, default=0.0)
    failure_count: int = field(init=False, default=0)

    def __post_init__(self):
        """Initializes the rate limiter and the API client after the dataclass is created."""
        if self.rpm_limit:
            self.limiter = AsyncLimiter(self.rpm_limit // 4, 15)
        self.client = instructor.from_openai(
            AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            ),
            mode=instructor.Mode.JSON,
        )

class LLMDispatcher:
    """
    Manages a pool of LLM clients, dispatching requests while respecting
    concurrency, per-minute, and persistent daily limits. Also handles
    exponential backoff and disabling of failing clients.
    """
    BASE_BACKOFF_DELAY = 10
    MAX_BACKOFF_SECONDS = 200

    def __init__(self, configs: List[LLMClientConfig], log_file: str = "daily_request_log.json"):
        self.clients = configs
        self.client_semaphores = {
            config.name: asyncio.Semaphore(config.concurrency) for config in configs
        }
        self._client_lock = asyncio.Lock()
        self.log_file = Path(log_file)
        
        # --- NEW: Add state for round-robin ---
        self._next_client_index = 0
        # --- END NEW ---
        
        self._load_daily_counts()
        logger.info(f"LLM Dispatcher initialized with {len(self.clients)} clients for round-robin usage.")
        for client in self.clients:
            logger.info(
                f"Client '{client.name}' loaded with daily count: "
                f"{client.daily_counter}/{client.daily_limit or 'Unlimited'}")
    
    def _get_today_str(self) -> str:
        return datetime.date.today().isoformat()

    def _load_daily_counts(self):
        today_str = self._get_today_str()
        data_for_today = {}

        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    full_data = json.load(f)
                if today_str in full_data:
                    data_for_today = {today_str: full_data[today_str]}
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not read or parse log file {self.log_file}. Starting fresh. Error: {e}")
                data_for_today = {today_str: {}}

        if today_str not in data_for_today:
            data_for_today[today_str] = {}

        today_counts = data_for_today.get(today_str, {})
        for client in self.clients:
            client.daily_counter = today_counts.get(client.name, 0)

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data_for_today, f, indent=4)
        except IOError as e:
            logger.error(f"Could not write initial log file {self.log_file}. Daily limits may not persist. Error: {e}")

    def _update_persistent_count(self, client: LLMClientConfig):
        today_str = self._get_today_str()
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {today_str: {}}

        if today_str not in data:
            data = {today_str: {}}

        data[today_str][client.name] = client.daily_counter

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to update persistent count for {client.name}. Error: {e}")

    async def release_client_on_failure(self, client: LLMClientConfig):
        async with self._client_lock:
            if client.is_disabled:
                return

            client.failure_count += 1
            failure_num = client.failure_count

            backoff_duration = self.BASE_BACKOFF_DELAY * (2 ** (failure_num - 1))

            if backoff_duration > self.MAX_BACKOFF_SECONDS:
                last_backoff = self.BASE_BACKOFF_DELAY * (2 ** (failure_num - 2))
                if last_backoff >= self.MAX_BACKOFF_SECONDS:
                    client.is_disabled = True
                    logger.critical(
                        f"Client '{client.name}' has been PERMANENTLY DISABLED due to repeated failures."
                    )
                    return

            backoff_duration = min(backoff_duration, self.MAX_BACKOFF_SECONDS)

            client.is_in_timeout = True
            client.timeout_until = time.time() + backoff_duration
            logger.warning(
                f"Client '{client.name}' failed. Timed out for {backoff_duration:.2f} seconds. "
                f"Consecutive failures: {client.failure_count}."
            )

    async def release_client_on_success(self, client: LLMClientConfig):
        async with self._client_lock:
            if client.failure_count > 0:
                logger.info(f"Client '{client.name}' succeeded. Resetting failure count from {client.failure_count} to 0.")
                client.failure_count = 0


    async def get_available_client(self) -> Optional[LLMClientConfig]:
        """
        Atomically selects an available client using a round-robin strategy
        to ensure all clients are utilized.
        """
        async with self._client_lock:
            # Date-check logic remains the same
            if self._get_today_str() not in self._read_log_file_safely():
                logger.info("New day detected. Resetting all daily counters.")
                for client in self.clients:
                    client.daily_counter = 0
                self._load_daily_counts()

            # --- MODIFIED: Implement Round-Robin Logic ---
            # Try to find a client for one full loop starting from the last position.
            for i in range(len(self.clients)):
                # Get the client index to check, wrapping around the list if necessary
                client_index = (self._next_client_index + i) % len(self.clients)
                client = self.clients[client_index]

                # 1. Check if permanently disabled
                if client.is_disabled:
                    continue

                # 2. Check if in timeout
                if client.is_in_timeout:
                    if time.time() >= client.timeout_until:
                        client.is_in_timeout = False
                        logger.info(f"Client '{client.name}' is no longer in timeout.")
                    else:
                        continue # Still in timeout

                # 3. Check daily limit
                if client.daily_limit is not None and client.daily_counter >= client.daily_limit:
                    continue
                
                # 4. Check per-minute limit (via concurrency semaphore)
                if self.client_semaphores[client.name].locked():
                    continue
                
                # --- Client is available. Select it. ---
                client.daily_counter += 1
                self._update_persistent_count(client)
                
                # Update the starting point for the *next* search to ensure rotation
                self._next_client_index = (client_index + 1) % len(self.clients)

                logger.info(
                    f"Assigned request to {client.name} (Round-Robin). "
                    f"Daily count: {client.daily_counter}/{client.daily_limit or 'Unlimited'}"
                )
                return client
            # --- END MODIFIED SECTION ---
            
            # logger.warning("No available LLM clients found after checking all possibilities.")
            return None

        """
        Atomically selects a client that is not disabled, in timeout,
        or over its usage limits.
        """
        # --- METRICS: Measure time spent inside the lock ---
        lock_start_time = time.time()
        async with self._client_lock:
            lock_duration = time.time() - lock_start_time
            if lock_duration > 0.01: # Log only if it's non-trivial
                 logger.info(f"METRIC: Waited {lock_duration:.4f}s to acquire client lock.")
            find_start_time = time.time()
            # Date-check logic
            if self._get_today_str() not in self._read_log_file_safely():
                logger.info("New day detected. Resetting all daily counters.")
                for client in self.clients:
                    client.daily_counter = 0
                self._load_daily_counts()

            for client in self.clients:
                if client.is_disabled:
                    continue

                if client.is_in_timeout:
                    if time.time() >= client.timeout_until:
                        client.is_in_timeout = False
                        client.timeout_until = 0.0
                        logger.info(f"Client '{client.name}' is no longer in timeout.")
                    else:
                        continue

                if client.daily_limit is not None and client.daily_counter >= client.daily_limit:
                    continue
                
                if client.limiter and not client.limiter.has_capacity():
                    continue
                
                # Client is available.
                client.daily_counter += 1
                self._update_persistent_count(client)
                logger.info(
                    f"Assigned request to {client.name}. "
                    f"Daily count: {client.daily_counter}/{client.daily_limit or 'Unlimited'}"
                )
                find_duration = time.time() - find_start_time
                logger.info(f"METRIC: Client finding logic (inside lock) took {find_duration:.4f}s")
                return client
            
            find_duration = time.time() - find_start_time
            logger.warning(f"METRIC: Client finding logic (inside lock) took {find_duration:.4f}s but found NO client.")
            logger.warning("No available LLM clients found.")
            return None
    
    def _read_log_file_safely(self) -> dict:
        """Safely reads the JSON log file, returning an empty dict on failure."""
        if not self.log_file.exists():
            return {}
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

def get_llm_configs() -> List[LLMClientConfig]:
    return [
        LLMClientConfig(
            name="Cerebras-Free-Llama-70b",
            model="llama-3.3-70b",
            base_url="https://api.cerebras.ai/v1",
            api_key=config.CEREBRAS_API_KEY,
            rpm_limit=15,
            daily_limit=700,
            concurrency=5
        ),
        LLMClientConfig(
            name="Cerebras-Free-Qwen3-32b",
            model="qwen-3-32b",
            base_url="https://api.cerebras.ai/v1",
            api_key=config.CEREBRAS_API_KEY,
            rpm_limit=15,
            daily_limit=250,
            concurrency=10
        ),
        LLMClientConfig(
            name="Cerebras-Free-Llama-Scout",
            model="llama-4-scout-17b-16e-instruct",
            base_url="https://api.cerebras.ai/v1",
            api_key=config.CEREBRAS_API_KEY,
            rpm_limit=15,
            daily_limit=750,
            concurrency=10
        ),
        LLMClientConfig(
            name="Cerebras-Free-Qwen3-235b",
            model="qwen-3-235b-a22b",
            base_url="https://api.cerebras.ai/v1",
            api_key=config.CEREBRAS_API_KEY,
            rpm_limit=15,
            daily_limit=350,
            concurrency=10
        ),
        LLMClientConfig(
            name="OpenRouter-Free-Qwen",
            model="qwen/qwen3-32b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPEN_ROUTER_API_KEY,
            rpm_limit=15,
            daily_limit=1000,
            concurrency=1
        ),
        LLMClientConfig(
            name="Magistral Medium",
            model="magistral-medium-2506",
            base_url="https://api.mistral.ai/v1/",
            api_key=config.MISTRAL_API_KEY,
            rpm_limit=40,
            daily_limit=100000,
            concurrency=10
        ),
        LLMClientConfig(
            name="Together",
            model="serverless-qwen-qwen3-32b-fp8",  
            base_url="https://api.together.xyz/v1",
            api_key=config.TOGETHER_API_KEY, 
            rpm_limit=15,
            daily_limit=100000,
            concurrency=1
        ),
        LLMClientConfig(
            name="Groq-maverick",
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            base_url="https://api.groq.com/openai/v1/",
            api_key=config.GROQ_API_KEY, 
            rpm_limit=10,
            daily_limit=10000,
            concurrency=1
        ),
        LLMClientConfig(
            name="Groq-scout",
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            base_url="https://api.groq.com/openai/v1/",
            api_key=config.GROQ_API_KEY,  
            rpm_limit=10,
            daily_limit=10000,
            concurrency=1
        ),
        LLMClientConfig(
            name="Groq-qwen32b",
            model="qwen/qwen3-32b",  # Example model, choose as needed
            base_url="https://api.groq.com/openai/v1/",
            api_key=config.GROQ_API_KEY,  # Replace with your Together API key variable
            rpm_limit= 5,
            daily_limit=10000,
            concurrency=10
        ),
        LLMClientConfig(
            name="Groq-QwQ",
            model="qwen-qwq-32b",  # Example model, choose as needed
            base_url="https://api.groq.com/openai/v1/",
            api_key=config.GROQ_API_KEY,  # Replace with your Together API key variable
            rpm_limit=5,
            daily_limit=10000,
            concurrency=1
        ),
        LLMClientConfig(
            name="OpenRouter-Mistral-Small-Paid",
            model="mistralai/mistral-small-3.2-24b-instruct",  # Example model, choose as needed
            base_url="https://openrouter.ai/api/v1",
            api_key=config.GROQ_API_KEY,  # Replace with your Together API key variable
            rpm_limit=60,
            daily_limit=10000,
            concurrency=1
        ),
        LLMClientConfig(
            name="OpenRouter-Mistral-Small-Free",
            model="mistralai/mistral-small-3.2-24b-instruct:free",  # Example model, choose as needed
            base_url="https://openrouter.ai/api/v1",
            api_key=config.GROQ_API_KEY,  # Replace with your Together API key variable
            rpm_limit=15,
            daily_limit=10000,
            concurrency=1
        ),
    ]