import os
import functools
from cls_luigi.utils.wandb.core import wandb_log_artifact, wandb_log_model


def log_output(output_config=None, callback=None):
    """Decorator to log Luigi task outputs to Weights & Biases with custom configuration.

    This decorator automatically logs the outputs of a Luigi task to Weights & Biases
    after the task has completed successfully. It patches the task's run method to
    perform logging only after all outputs have been written, ensuring data integrity.

    Args:
        output_config: Configuration for how outputs should be logged to W&B:
            - None: Use default logging (type="output")
            - str: Specify type for single output (e.g., "dataset", "model")
            - Dict: Full configuration for single output with type, metadata, etc.
            - List: List of configurations for list outputs (must match output list length)
            - Dict of configs: Mapping output names to their configurations for dict outputs
        callback: Optional function to execute after outputs are logged. The function
                 will receive the task outputs as its argument.

    Returns:
        Decorated output method that handles W&B logging after task completion.

    Examples:
        ```python
        # Default logging with default type
        @log_output
        def output(self):
            return luigi.LocalTarget("output.csv")

        # Default logging with default type
        @log_output()
        def output(self):
            return luigi.LocalTarget("output.csv")

        # Specify artifact type for single output
        @log_output("dataset")
        def output(self):
            return luigi.LocalTarget("data.csv")

        # Full configuration for single output
        @log_output({
            "type": "model",
            "metadata": {"accuracy": 0.95},
            "aliases": ["best", "v1"]
        })
        def output(self):
            return luigi.LocalTarget("model.pkl")

        # Configuration for list outputs
        @log_output([
            "dataset",
            {"type": "plot", "metadata": {"metric": "loss"}}
        ])
        def output(self):
            return [
                luigi.LocalTarget("data.csv"),
                luigi.LocalTarget("plot.png")
            ]

        # Configuration for dictionary outputs
        @log_output({
            "data": "dataset",
            "visualization": {"type": "plot", "metadata": {"metric": "accuracy"}}
        })
        def output(self):
            return {
                "data": luigi.LocalTarget("data.csv"),
                "visualization": luigi.LocalTarget("plot.png")
            }
        ```

    Note:
        For model artifacts (type="model"), the decorator will automatically use
        wandb_log_model instead of wandb_log_artifact for proper model versioning.
        It will also automatically set the use_as parameter based on common artifact
        types if not explicitly specified.
    """
    # Handle the case when decorator is used without arguments
    if callable(output_config) and callback is None:
        func = output_config
        output_config = None
        return log_output(output_config)(func)

    def decorator(output_method):
        @functools.wraps(output_method)
        def wrapper(self, *args, **kwargs):
            # Get the output target(s)
            targets = output_method(self, *args, **kwargs)

            # Store the original targets in task instance for later use
            if not hasattr(self, "_output_targets"):
                self._output_targets = targets

            # Patch the run method if not already patched
            if not hasattr(self, "_run_patched"):
                original_run = self.run

                # Track if logging has been executed
                self._logging_executed = False

                @functools.wraps(original_run)
                def patched_run(*run_args, **run_kwargs):
                    # Call the original run method
                    result = original_run(*run_args, **run_kwargs)

                    # After run completes, check if outputs exist using Luigi's method
                    # Only execute logging if it hasn't been executed before
                    if not self._logging_executed:
                        # Use Luigi's complete() method which checks if all outputs exist
                        if self.complete():
                            self._logging_executed = True
                            outputs = self._output_targets

                            # Log outputs to wandb
                            _log_to_wandb(self, outputs, output_config)

                            # Execute additional callback if provided
                            if callback:
                                callback(outputs)

                    return result

                # Replace the run method
                self.run = patched_run
                self._run_patched = True

            return targets

        return wrapper

    return decorator


def _log_to_wandb(task, outputs, output_config=None):
    """Helper function to log Luigi task outputs to Weights & Biases with specified configurations.

    This internal function handles the actual logging of Luigi task outputs to W&B,
    creating properly configured artifacts for versioning and tracking. It supports
    various output formats (single, list, dictionary) and configuration options.

    For model artifacts (type="model"), it automatically uses wandb_log_model instead
    of wandb_log_artifact for proper model versioning and tracking. It also automatically
    sets the use_as parameter based on common artifact types if not explicitly specified.

    Args:
        task: The Luigi task instance that generated the outputs
        outputs: The output targets to log (single target, list of targets, or dict of targets)
        output_config: Configuration for how outputs should be logged:
            - None: Use default logging (type="output")
            - str: Specify type for single output (e.g., "dataset", "model")
            - Dict: Full configuration for single output with type, metadata, etc.
            - List: List of configurations for list outputs (must match output list length)
            - Dict: Mapping output names to their configurations for dict outputs

    Raises:
        ValueError: If the configuration list length doesn't match the outputs list length

    Note:
        This function automatically adds task metadata to the artifact, including task_id
        and task_family. It also supports dynamic metadata through callable values that
        receive the task instance as an argument.
    """

    def prepare_config(config, default_type="output"):
        """Normalize configuration input to a standard dictionary format.

        Args:
            config: Configuration input (string type or dictionary)
            default_type: Default artifact type to use if none specified

        Returns:
            Dictionary containing the normalized configuration
        """
        # Convert simple string type to full config
        if isinstance(config, str):
            return {"type": config}
        # Use default if no config provided
        return config or {"type": default_type}

    def log_artifact(output, name=None, config=None):
        """Log a single Luigi target as a W&B artifact.

        Args:
            output: Luigi target to log
            name: Name for the artifact (defaults to output string representation)
            config: Configuration for the artifact (type, metadata, etc.)
        """
        # Prepare configuration
        config = prepare_config(config)

        # Extract configuration
        artifact_type = config.get("type", "output")
        metadata = config.get("metadata", {})
        description = config.get("description", "")
        aliases = config.get("aliases", ["latest"])

        # Add default metadata
        metadata.setdefault("file_count", 1)
        try:
            metadata.setdefault("file_size", os.path.getsize(output.path))
        except Exception:
            pass

        # Add task information to metadata
        metadata.setdefault(
            "task_id", task.task_id if hasattr(task, "task_id") else None
        )
        metadata.setdefault("task_family", task.__class__.__name__)

        # Handle dynamic metadata through lambda functions
        processed_metadata = {}
        for key, value in metadata.items():
            if callable(value) and hasattr(value, "__call__"):
                try:
                    processed_metadata[key] = value(task)
                except Exception as e:
                    processed_metadata[key] = f"Error evaluating: {str(e)}"
            else:
                processed_metadata[key] = value

        # Auto-set use_as based on type if not explicitly provided
        use_as = config.get("use_as", None)
        if use_as is None:
            # Common artifact types that have natural use_as values
            type_to_use_as = {
                "model": "model",
                "dataset": "dataset",
                "config": "config",
                "checkpoint": "checkpoint",
            }
            use_as = type_to_use_as.get(artifact_type)

        # Create artifact configuration
        artifact_config = {
            "path": output.path,
            "type": artifact_type,
            "metadata": processed_metadata,
            "description": description,
            "aliases": aliases,
            "tags": config.get("tags", []),
            "incremental": config.get("incremental", False),
            "use_as": use_as,
        }

        artifact_name = str(name or output.__str__())

        # Use wandb_log_model for model artifacts, wandb_log_artifact for others
        if artifact_type == "model":
            wandb_log_model({artifact_name: artifact_config})
        else:
            # Log the artifact using the enhanced wandb_log_artifact function
            wandb_log_artifact({artifact_name: artifact_config})

    # Handle different output types with configuration
    if isinstance(outputs, list):
        # List outputs with optional list config
        if isinstance(output_config, list):
            # Ensure config list matches outputs list
            if len(output_config) != len(outputs):
                raise ValueError("Configuration list must match output list length")
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
