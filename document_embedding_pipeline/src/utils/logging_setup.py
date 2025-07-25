import logging
from typing import Dict, Any
from .config_loader import ConfigLoader

def get_component_logger(module_name: str) -> logging.Logger:
    """
    Creates and configures a logger for a specific component based on module path.
    
    Args:
        module_name: The __name__ of the module where the logger is being created
        
    Returns:
        Configured logger instance for the component
    """
    # Remove base package name
    component = module_name.replace('document_embedding_pipeline.src.', '')
    component = component.replace(".", "_")

    if component == "__main__":
        component = "main"
    
    # Get logging configuration using existing ConfigLoader
    try:
        # Use ConfigLoader to get the entire config
        config = ConfigLoader.load_config()
        logging_config = config.get('logging', {})
    except RuntimeError as e:
        # Fallback to basic logging if config loading fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(component)
        logger.error(f"Failed to load configuration: {e}. Using default logging settings.")
        return logger
    
    # Set up logger
    logger = logging.getLogger(component)
    
    # Set log level
    global_level = logging_config.get('global_level', 'INFO')
    component_level = logging_config.get('components', {}).get(component, global_level)
    logger.setLevel(component_level)
    
    # Create console handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs from root logger    
    return logger
