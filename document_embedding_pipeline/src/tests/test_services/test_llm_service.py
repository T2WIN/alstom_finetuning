import logging
from pydantic import Field, BaseModel
from services.llm_service import LLMService
from openai import OpenAIError

# 1. Configure basic logging to see the output from the service.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 2. Define a simple Pydantic model for the test.
#    This schema will be used to validate the LLM's output.
class UserInfo(BaseModel):
    name: str = Field(description="The full name of the user.")
    age: int = Field(description="The age of the user.")
    role: str = Field(description="The job title or role of the user.")

# 3. The prompt to send to the language model.
test_prompt = "Extract the user info for a 34-year-old software engineer named John Doe."

# 4. The model to use for the test.
#    You can change this to any model available on your Ollama server.
test_model = "Qwen3-14B-Q5_K_M.gguf"

logger.info("--- Starting LLMService Test ---")
logger.info(f"Using model: {test_model}")
# logger.info(f"Prompt: '{test_prompt}'")

try:
    # 5. Initialize the LLMService.
    #    This will set up the connection to the Ollama server.
    llm_service = LLMService()

    # 6. Call the service to get a structured response.
    #    The service will send the prompt and expect a JSON response that
    #    conforms to the UserInfo schema.
    user_info = llm_service.get_structured_response(
        prompt=test_prompt,
        model_name=test_model,
        response_model=UserInfo,
    )

    # 7. Print the results.
    #    If successful, this will display the Pydantic object.
    logger.info("--- Test Successful ---")
    logger.info(f"Received structured data:\n{user_info.model_dump_json(indent=2)}")

except OpenAIError as e:
    logger.error(
        f"--- Test Failed: Could not connect to Ollama or model '{test_model}' not found. ---"
    )
    logger.error(f"Please ensure Ollama is running and the model is pulled (e.g., 'ollama pull {test_model}').")
    logger.error(f"Details: {e}")
except Exception as e:
    logger.error(f"--- Test Failed: An unexpected error occurred. ---")
    logger.error(f"Details: {e}", exc_info=True)