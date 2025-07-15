# test_logging.py

import logging
import os
import shutil
import time
from utils.logging_setup import setup_logging
# --- Mock functions to simulate a real application ---

def simulate_data_processing():
    """Simulates a function from another module."""
    # Get a logger instance specific to this "module"
    logger = logging.getLogger("data_processor")
    logger.info("Starting data processing task.")
    logger.debug("Checking data integrity...")
    time.sleep(0.5)
    logger.debug("Data integrity check passed.")
    logger.warning("A non-critical value is missing. Using default.")
    logger.info("Data processing task finished.")

def simulate_pipeline_failure():
    """Simulates a critical failure."""
    logger = logging.getLogger("pipeline.core")
    logger.error("Failed to connect to required service.")
    try:
        raise ConnectionError("Could not establish connection to database.")
    except ConnectionError:
        logger.critical(
            "Unrecoverable error: Database connection failed. Shutting down.",
            exc_info=True # exc_info=True logs the full exception traceback
        )

# --- Main test execution ---

def main():
    """Main function to run the test."""
    output_dir = "temp_test_logs"
    
    print("--- TEST 1: Running with INFO level ---")
    # Setup logging with INFO level for the console
    setup_logging(output_folder=output_dir, log_level="INFO")
    
    # Get a logger for the main script
    main_logger = logging.getLogger("main_test")
    
    main_logger.debug("This is a debug message. It should NOT appear on the console.")
    main_logger.info("Starting the test script.")
    
    simulate_data_processing()
    simulate_pipeline_failure()
    
    main_logger.info("Test script finished.")
    
    print(f"\nLog file has been created at: {os.path.join(output_dir, 'pipeline.log')}")
    print("Please inspect the file to see all log messages (including DEBUG).")
    print("\n--- End of TEST 1 ---\n")
    
    # --- Second test with DEBUG level ---
    print("--- TEST 2: Running with DEBUG level ---")
    setup_logging(output_folder=output_dir, log_level="DEBUG")
    debug_logger = logging.getLogger("main_test_debug")
    debug_logger.debug("This debug message SHOULD now appear on the console.")
    debug_logger.info("Test 2 complete.")
    print("\n--- End of TEST 2 ---\n")


if __name__ == "__main__":
    main()
    
    # --- Cleanup ---
    # In a real run you wouldn't delete this, but for a clean test we do.
    val = input("Press Enter to delete the 'temp_test_logs' and 'src' directories...")
    if os.path.exists("temp_test_logs"):
        shutil.rmtree("temp_test_logs")
        print("Cleaned up 'temp_test_logs' directory.")


