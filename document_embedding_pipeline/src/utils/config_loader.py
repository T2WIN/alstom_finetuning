# src/utils/config_loader.py

import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """Utility class for loading and accessing configuration values."""
    
    _config: Dict[str, Any] = None
    
    @classmethod
    def load_config(cls, config_path: Path = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if cls._config is None:
            if config_path is None:
                config_path = Path("config.yaml")
            
            try:
                with open(config_path, 'r', encoding="utf-8") as f:
                    cls._config = yaml.safe_load(f)
            except FileNotFoundError:
                raise RuntimeError(f"Configuration file not found: {config_path}")
        
        return cls._config
    
    @classmethod
    def get(cls, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation (e.g., 'llm.models.primary_model')."""
        if cls._config is None:
            cls.load_config()
        
        keys = key_path.split('.')
        value = cls._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key_path}")