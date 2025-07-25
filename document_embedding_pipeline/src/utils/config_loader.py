from __future__ import annotations
import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, List
from pydantic import BaseModel, ValidationError

# Pydantic models for configuration validation
class PathsConfig(BaseModel):
    embedding_model_dir: str
    artifacts_path: str
    temp_dir_name: str
    failed_dir_name: str
    qdrant_db_path: str
    log_file: str

class LLMConfig(BaseModel):
    ollama_base_url: str
    tokenizer_model: str
    max_input_tokens: int

class QdrantConfig(BaseModel):
    collection_name: str
    vector_size: int
    distance_metric: str

class WordProcessingParams(BaseModel):
    small_doc_threshold_tokens: int
    title_model: str
    structure_model: str
    summary_model: str

class ExcelProcessingParams(BaseModel):
    table_truncation_rows: int

class ProcessingParamsConfig(BaseModel):
    max_retries: int
    word: WordProcessingParams
    excel: ExcelProcessingParams

class UnoserverConfig(BaseModel):
    host: str
    port: int

class ServicesConfig(BaseModel):
    unoserver: UnoserverConfig

class LoggingComponentConfig(BaseModel):
    pipeline: Optional[str] = None
    main: Optional[str] = None
    pipeline_word_processor: Optional[str] = None
    pipeline_excel_processor: Optional[str] = None
    services: Optional[str] = None
    services_llm_service: Optional[str] = None
    services_docling_service: Optional[str] = None
    services_unoserver_service: Optional[str] = None
    services_qdrant_service: Optional[str] = None
    utils: Optional[str] = None
    utils_state_manager: Optional[str] = None

class LoggingConfig(BaseModel):
    global_level: str
    components: LoggingComponentConfig

class AppConfig(BaseModel):
    paths: PathsConfig
    llm: LLMConfig
    qdrant: QdrantConfig
    processing_params: ProcessingParamsConfig
    services: ServicesConfig
    logging: LoggingConfig

    class Config:
        extra = "allow"

class ConfigLoader:
    """Utility class for loading and accessing configuration values with Pydantic validation."""
    
    _config: Dict[str, Any] = None

    @classmethod
    def load_config(cls, config_path: Path = None) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file with Pydantic validation."""
        if cls._config is None:
            if config_path is None:
                # First try config.yaml, then config.json for backward compatibility
                config_path = Path("config.yaml")
                if not config_path.exists():
                    config_path = Path("config.json")
            
            try:
                # Read file content
                with open(config_path, 'r', encoding="utf-8") as f:
                    if config_path.suffix.lower() == ".json":
                        raw_data = json.load(f)
                    else:
                        raw_data = yaml.safe_load(f)
                
                # Apply environment variable overrides for all parameters
                cls._apply_environment_overrides(raw_data)
                
                # Validate with Pydantic
                app_config = AppConfig(**raw_data)
                cls._config = app_config.model_dump()
            
                
                # Convert to JSON and back to ensure consistent structure
                json_data = json.dumps(cls._config)
                cls._config = json.loads(json_data)
                
            except FileNotFoundError:
                raise RuntimeError(f"Configuration file not found: {config_path}")
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON configuration: {e}")
            except yaml.YAMLError as e:
                raise RuntimeError(f"Invalid YAML configuration: {e}")
            except ValidationError as e:
                error_details = "\n".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
                raise RuntimeError(
                    f"Configuration validation error:\n{error_details}\n"
                    "Please check your configuration file structure and values."
                )
            except ValueError as e:
                raise RuntimeError(f"Configuration error: {str(e)}")
        
        return cls._config

    @classmethod
    def _apply_environment_overrides(cls, raw_data: dict):
        """Override config values with environment variables for any parameter"""
        def recursive_override(data: dict, parent_key: str = ""):
            for key, value in data.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, dict):
                    recursive_override(value, full_key)
                else:
                    env_var = full_key.replace('.', '_').upper()
                    if env_var in os.environ:
                        data[key] = os.environ[env_var]
        
        recursive_override(raw_data)

    @classmethod
    def _get_nested_value(cls, data: dict, keys: List[str]) -> Any:
        """Recursively get nested value from dictionary"""
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Key '{key}' not found in configuration path")
            current = current[key]
        return current

    @classmethod
    def get(cls, key_path: str, default: Optional[Any] = None) -> Any:
        """Get configuration value using dot notation with enhanced error handling"""
        if cls._config is None:
            cls.load_config()
        
        keys = key_path.split('.')
        
        try:
            return cls._get_nested_value(cls._config, keys)
        except KeyError as e:
            if default is not None:
                return default
            raise KeyError(
                f"Configuration key '{key_path}' not found. "
                f"Available top-level keys: {', '.join(cls._config.keys())}"
            )
        except TypeError as e:
            if default is not None:
                return default
            raise TypeError(
                f"Invalid configuration access for path '{key_path}': {str(e)}"
            )