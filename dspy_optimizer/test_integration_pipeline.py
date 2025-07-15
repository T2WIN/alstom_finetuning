# test_pipeline_integration.py

import pytest
import dspy
import asyncio
import nest_asyncio
import logging
import time

# Import the full pipeline and its dependencies
from dspy_pipeline import SyntheticQuestionPipeline
from custom_lm import DispatcherLM # Make sure custom_lm.py has the provided logic
from llm_services import LLMDispatcher, get_llm_configs

# ... (logging and nest_asyncio setup remain the same) ...
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nest_asyncio.apply()


# 💡 FIX 1: Create a module-scoped event loop fixture.
# This ensures all tests in this module share the same loop.
@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the entire module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# 💡 FIX 2: Inject the event_loop into the pipeline fixture and pass it to the LM.
@pytest.fixture(scope="module")
def real_pipeline(): # 💡 FIX: No longer needs the event_loop fixture.
    """
    Sets up the full, un-mocked DSPy pipeline for integration testing.
    """
    dispatcher = LLMDispatcher(get_llm_configs())
    
    # 💡 FIX: Do NOT pass the loop to DispatcherLM.
    dspy_lm = DispatcherLM(dispatcher)
    
    dspy.settings.configure(lm=dspy_lm, rm=None)
    pipeline = SyntheticQuestionPipeline()
    return pipeline

@pytest.fixture
def sample_test_data():
    """
    Provides a small, representative sample of data for the integration test.
    """
    examples = [
        dspy.Example(
            chunk_text="The system shall ensure that the braking distance at a speed of 160 km/h does not exceed 1200 meters. This is calculated based on standard rail adhesion conditions and a fully loaded train.",
            persona_description="A safety engineer verifying compliance with European railway standards."
        ).with_inputs("chunk_text", "persona_description"),
        dspy.Example(
            chunk_text="Our new seat design incorporates lumbar support and uses fire-retardant Alcantara fabric, providing superior comfort and safety for long-distance journeys.",
            persona_description="A product manager focused on passenger experience and comfort features."
        ).with_inputs("chunk_text", "persona_description"),
        dspy.Example(
            chunk_text="The HVAC system is designed to maintain a cabin temperature of 22°C ± 2°C and circulates air at a rate of 1,500 cubic meters per hour, minimizing CO2 concentration.",
            persona_description="A maintenance technician troubleshooting environmental control systems."
        ).with_inputs("chunk_text", "persona_description"),
        dspy.Example(
            chunk_text="The train's diagnostic software logs over 5,000 data points per minute, including motor temperature, voltage fluctuations, and door sensor statuses. These logs are stored for 90 days for analysis.",
            persona_description="A data scientist looking for anomalies in operational performance data."
        ).with_inputs("chunk_text", "persona_description"),
        dspy.Example(
            chunk_text="To replace the primary inverter, first, disconnect the main battery using the certified safety switch. Then, unbolt the four main fasteners and carefully detach the coolant hoses.",
            persona_description="A field service engineer performing a critical component replacement."
        ).with_inputs("chunk_text", "persona_description"),
    ]
    # To really test concurrency, we can increase the number of tasks
    return examples * 2 # Creates 10 tasks in total


async def run_single_pipeline(pipeline: SyntheticQuestionPipeline, example: dspy.Example, task_id: int):
    """
    A helper coroutine to run a single pipeline execution and log its timing.
    """
    logger.info("[Task %d] Starting execution.", task_id)
    task_start_time = time.time()
    
    loop = asyncio.get_running_loop()
    prediction = await loop.run_in_executor(
        None,
        pipeline,
        example.chunk_text,
        example.persona_description
    )
    
    task_end_time = time.time()
    duration = task_end_time - task_start_time
    logger.info("[Task %d] Finished execution in %.2f seconds.", task_id, duration)
    logger.info("[Task %d] Generated Query: %s", task_id, prediction.query)
    
    # Assertions for this single task
    assert isinstance(prediction, dspy.Prediction), f"[Task {task_id}] The output should be a dspy.Prediction object."
    assert hasattr(prediction, 'query'), f"[Task {task_id}] The prediction should have a 'query' attribute."
    assert isinstance(prediction.query, str), f"[Task {task_id}] The generated query should be a string."
    assert len(prediction.query.strip()) > 0, f"[Task {task_id}] The generated query should not be empty."
    
    return prediction


@pytest.mark.asyncio
async def test_pipeline_full_integration(real_pipeline, sample_test_data):
    """
    Runs an end-to-end integration test on the SyntheticQuestionPipeline,
    executing all sample tests concurrently to verify asynchronous behavior.
    """
    logger.info("--- Starting Full Pipeline Integration Test with %d concurrent tasks ---", len(sample_test_data))
    
    start_time = time.time()

    # Create a list of concurrent tasks
    tasks = [
        run_single_pipeline(real_pipeline, example, i + 1)
        for i, example in enumerate(sample_test_data)
    ]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info("--- All %d tasks completed. ---", len(sample_test_data))
    logger.info("Total concurrent execution time: %.2f seconds.", total_duration)

    # A simple assertion to verify concurrency. 
    # If 10 tasks took less than 5 seconds, they almost certainly ran in parallel,
    # assuming a single API call takes more than 0.5 seconds on average.
    # Adjust this threshold based on your typical API response times.
    assert total_duration < (len(sample_test_data) * 0.8), "Total execution time suggests tasks did not run concurrently."
    assert len(results) == len(sample_test_data), "Did not receive results for all tasks."

