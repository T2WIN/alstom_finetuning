# test_debug_step2.py
import asyncio
import logging
import pytest
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DEBUG-STEP2")

def blocking_function_in_thread():
    """
    A synchronous function that runs in a separate thread.
    It will try to access the event loop.
    """
    logger.info("  [Thread] Now running in a separate thread.")
    
    thread_loop_id = None
    try:
        # Thanks to nest_asyncio, this should not raise an error and
        # should return the main thread's event loop.
        loop_from_thread = asyncio.get_running_loop()
        thread_loop_id = id(loop_from_thread)
        logger.info(f"  [Thread] ✅ Successfully accessed an event loop from the thread.")
        logger.info(f"  [Thread] ➡️ Event Loop ID seen from thread: {thread_loop_id}")
    except RuntimeError as e:
        logger.error(f"  [Thread] ❌ FAILED to get an event loop from the thread. Error: {e}")
    
    return thread_loop_id

@pytest.mark.asyncio
async def test_thread_event_loop_access():
    """
    This test checks if a separate thread can see the main event loop.
    The IDs logged below MUST be identical for your application to work.
    """
    logger.info("--- STARTING STEP 2: Threading Bridge Check ---")
    
    main_loop = asyncio.get_running_loop()
    main_loop_id = id(main_loop)
    logger.info(f"[Main] ➡️ Main Event Loop ID: {main_loop_id}")
    
    # Run the synchronous function in a separate thread using run_in_executor.
    with ThreadPoolExecutor(max_workers=1) as executor:
        thread_loop_id = await main_loop.run_in_executor(
            executor, blocking_function_in_thread
        )

    logger.info(f"[Main] The thread has finished execution.")
    
    if thread_loop_id is not None:
        logger.info(f"[Main] Comparing IDs: Main Loop ({main_loop_id}) vs. Thread Loop ({thread_loop_id})")
        assert main_loop_id == thread_loop_id, "CRITICAL ERROR: The event loop seen by the thread is DIFFERENT from the main loop!"
        logger.info("✅ SUCCESS: The thread correctly accessed the main event loop.")
    else:
        assert False, "The thread failed to access any event loop."

    logger.info("--- FINISHED STEP 2 ---")