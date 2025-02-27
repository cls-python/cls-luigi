import os
import functools
from cls_luigi.utils.wandb.core import wandb_log


def log_output(output_config=None):
    """
    Decorator to log outputs with optional custom configuration

    :param output_config:
    - None: use default logging
    - Dict: specify types/metadata for dictionary outputs
    - List: specify types/metadata for list outputs
    - Single dict: specify type/metadata for single output

    Usage examples:
    @log_output()  # Default logging
    @log_output('dataset')  # Specify type for single output
    @log_output({'type': 'dataset', 'metadata': {...}})  # Full config for single output
    @log_output([
        'dataset',
        {'type': 'plot', 'metadata': {...}}
    ])  # Config for list outputs
    @log_output({
        'demand': 'dataset',
        'plot': {'type': 'plot', 'metadata': {...}}
    })  # Config for dict outputs
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Call the original run method
            result = func(self, *args, **kwargs)

            # Handle different output structures
            outputs = self.output()

            def prepare_config(config, default_type="file"):
                # Convert simple string type to full config
                if isinstance(config, str):
                    return {"type": config}
                # Use default if no config provided
                return config or {"type": default_type}

            def log_artifact(output, name=None, config=None):
                # Prepare configuration
                config = prepare_config(config)

                # Extract configuration
                artifact_type = config.get("type", "file")
                metadata = config.get("metadata", {})

                # Add default metadata
                metadata.setdefault("file_count", 1)
                try:
                    metadata.setdefault("file_size", os.path.getsize(output.path))
                except Exception:
                    pass

                # Log the artifact
                wandb_log(
                    {
                        str(name or output.__str__()): {
                            "path": output.path,
                            "type": artifact_type,
                            "metadata": metadata,
                        }
                    }
                )

            # Handle different output types with configuration
            if isinstance(outputs, list):
                # List outputs with optional list config
                if isinstance(output_config, list):
                    # Ensure config list matches outputs list
                    if len(output_config) != len(outputs):
                        raise ValueError(
                            "Configuration list must match output list length"
                        )
                    for output, config in zip(outputs, output_config):
                        log_artifact(output, config=config)
                else:
                    # Use same config for all or default
                    for output in outputs:
                        log_artifact(output, config=output_config)

            elif isinstance(outputs, dict):
                # Dictionary outputs with optional dict config
                if isinstance(output_config, dict):
                    for key, output in outputs.items():
                        # Get config for this key, default to file
                        config = output_config.get(key, {"type": "file"})
                        log_artifact(output, name=key, config=config)
                else:
                    # Use same config for all or default
                    for key, output in outputs.items():
                        log_artifact(output, name=key, config=output_config)

            else:
                # Single output
                log_artifact(outputs, config=output_config)

            return result

        return wrapper

    return decorator
