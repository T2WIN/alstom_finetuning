# test_custom_lm.py

import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
from openai import APIConnectionError

# Import the classes to be tested and mocked
from custom_lm import DispatcherLM
from llm_services import LLMDispatcher, LLMClientConfig

# Configure logging to display INFO level messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# All test coroutines will be treated as marked with @pytest.mark.asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def mock_client_config(mocker):
    """
    Creates a mock LLMClientConfig.
    The underlying raw_client's API call method is replaced with an AsyncMock
    to prevent actual network calls.
    """
    config = LLMClientConfig(
        name="test-client",
        model="test-model",
        base_url="http://localhost:1234/v1",
        api_key="fake-key",
        rpm_limit=100,
        concurrency=2
    )
    
    # Mock the actual API call method on the raw_client
    mock_completion_create = AsyncMock()
    
    # To simulate the response structure from OpenAI's client
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a mock LLM response."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_completion_create.return_value = mock_response

    mocker.patch.object(config.raw_client.chat.completions, 'create', mock_completion_create)
    
    return config


@pytest.fixture
def mock_dispatcher(mock_client_config, mocker):
    """
    Creates an LLMDispatcher instance with the mocked client config.
    We can spy on its methods to ensure they are called correctly.
    """
    dispatcher = LLMDispatcher(configs=[mock_client_config])

    # Spy on the release methods to check if they get called
    mocker.spy(dispatcher, 'release_client_on_success')
    mocker.spy(dispatcher, 'release_client_on_failure')
    
    return dispatcher


@pytest.fixture
def dispatcher_lm(mock_dispatcher):
    """
    Provides an instance of our custom DispatcherLM, configured with the mock dispatcher.
    
    Args:
        mock_dispatcher: The mocked LLMDispatcher instance.
        event_loop: The asyncio event loop provided by pytest-asyncio.
    """
    # *** FIX: Pass the event_loop to the DispatcherLM constructor. ***
    return DispatcherLM(dispatcher=mock_dispatcher)


@pytest.mark.asyncio
async def test_successful_arequest(dispatcher_lm, mock_dispatcher):
    """
    Tests a successful asynchronous request through the DispatcherLM.
    """
    prompt = "Tell me a joke."
    completions = await dispatcher_lm.arequest(prompt)

    assert completions == ["This is a mock LLM response."]
    mock_dispatcher.release_client_on_success.assert_called_once()
    mock_dispatcher.release_client_on_failure.assert_not_called()


@pytest.mark.asyncio
async def test_arequest_failure_handling(dispatcher_lm, mock_dispatcher, mock_client_config):
    """
    Tests the failure handling mechanism.
    """
    # Arrange
    # Configure the mock API call on the raw_client to raise an error
    api_error = APIConnectionError(request=MagicMock(), message="Simulated connection error.")
    mock_client_config.raw_client.chat.completions.create.side_effect = api_error

    # Act & Assert
    # We expect our custom LM to catch the specific error and re-raise it.
    with pytest.raises(APIConnectionError, match="Simulated connection error."):
        await dispatcher_lm.arequest("This will fail.")

    # Assert that the dispatcher's failure method was called
    mock_dispatcher.release_client_on_failure.assert_called_once()
    mock_dispatcher.release_client_on_success.assert_not_called()


@pytest.mark.asyncio
async def test_no_client_available(dispatcher_lm, mock_dispatcher, mocker):
    """
    Tests the scenario where the dispatcher cannot provide an available client.
    """
    # Arrange
    mocker.patch.object(mock_dispatcher, 'get_available_client', new_callable=AsyncMock, return_value=None)

    # Act & Assert
    with pytest.raises(RuntimeError, match="No available LLM clients found."):
        await dispatcher_lm.arequest("No one can hear me.")


def test_synchronous_call_integration(dispatcher_lm, mock_dispatcher):
    """
    Tests the synchronous __call__ method.
    """
    prompt = "This is a synchronous call."
    completions = dispatcher_lm(prompt)

    assert completions == ["This is a mock LLM response."]
    mock_dispatcher.release_client_on_success.assert_called_once()
    mock_dispatcher.release_client_on_failure.assert_not_called()
