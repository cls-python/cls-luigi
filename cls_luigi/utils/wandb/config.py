from typing import Dict, Any, Optional, Type
import wandb
from .core import _validate_initialization

# Optional Pydantic imports with fallback
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # Type placeholder
    ValidationError = Exception  # Fallback exception type

def update_config(config: Dict[str, Any]) -> None:
    """Update the W&B config after initialization
    
    Args:
        config: Dictionary of configuration parameters to update
    """
    _validate_initialization()
    wandb.config.update(config)

def validate_config(
    config: Dict[str, Any],
    schema: Optional[Dict[str, type]] = None,
    pydantic_model: Optional[Type[BaseModel]] = None
) -> bool:
    """Validate configuration against a schema with optional Pydantic support
    
    Args:
        config: Configuration dictionary to validate
        schema: Simple type schema dictionary
        pydantic_model: Pydantic model for validation (requires pydantic install)
        
    Returns:
        bool: True if validation succeeds
        
    Raises:
        ImportError: If pydantic_model is used without pydantic installed
        ValueError: For validation failures
    """
    if pydantic_model:
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic validation requires pydantic package. "
                "Install with: pip install pydantic"
            )
            
        try:
            pydantic_model(**config)
            return True
        except ValidationError as e:
            raise ValueError(f"Config validation failed: {e}") from e
            
    if schema:
        for key, expected_type in schema.items():
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
            if not isinstance(config[key], expected_type):
                raise TypeError(
                    f"Config key '{key}' should be {expected_type}, "
                    f"got {type(config[key])}"
                )
        return True
    
    raise ValueError("Must provide either schema or pydantic_model")