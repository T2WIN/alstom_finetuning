# test_debug_step1.py
import asyncio
import logging
import pytest
import nest_asyncio

# Apply nest_asyncio at the very top, as in your original test.
nest_asyncio.apply()

# Configure logging to be visible.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DEBUG-STEP1")

@pytest.mark.asyncio
async def test_baseline_event_loop():
    """
    This test establishes the ID of the main event loop provided by pytest-asyncio.
    """
    logger.info("--- STARTING STEP 1: Baseline Environment Check ---")
    
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"✅ Successfully retrieved the running event loop.")
        logger.info(f"➡️ Baseline Event Loop ID: {id(loop)}")
    except RuntimeError as e:
        logger.error(f"❌ FAILED to get a running event loop. Error: {e}")
    
    logger.info("--- FINISHED STEP 1 ---")

    # This assertion is just to make the test pass.
    assert True