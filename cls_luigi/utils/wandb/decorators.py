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

    if output_config is not None:
        return decorator(output_config)
    return decorator


def after_output_written(func=None, callback=None):
    """
    Decorator for Luigi Task.output method that executes a callback 
    only after the output has been written.
    
    The callback will be called with the output target(s) as argument.
    
    Usage:
    
    @after_output_written
    def output(self):
        return luigi.LocalTarget("myfile.txt")
        
    # Or with a custom callback
    @after_output_written(callback=lambda target: print(f"Output written to {target.path}"))
    def output(self):
        return luigi.LocalTarget("myfile.txt")
    """
    def decorator(output_method):
        @functools.wraps(output_method)
        def wrapper(self, *args, **kwargs):
            # Get the output target(s)
            targets = output_method(self, *args, **kwargs)
            
            # Store the original targets in task instance for later use
            if not hasattr(self, '_output_targets'):
                self._output_targets = targets
            
            # Patch the run method if not already patched
            if not hasattr(self, '_run_patched'):
                original_run = self.run
                
                @functools.wraps(original_run)
                def patched_run(*run_args, **run_kwargs):
                    # Call the original run method
                    result = original_run(*run_args, **run_kwargs)
                    
                    # After run completes, check if outputs exist
                    outputs = self._output_targets
                    
                    # Function to check if a target exists
                    def target_exists(target):
                        try:
                            return target.exists()
                        except AttributeError:
                            # For targets that don't support exists(), check if path exists
                            try:
                                return os.path.exists(target.path)
                            except (AttributeError, TypeError):
                                return False
                    
                    # Check all outputs
                    all_exist = False
                    if isinstance(outputs, dict):
                        all_exist = all(target_exists(target) for target in outputs.values())
                    elif isinstance(outputs, list):
                        all_exist = all(target_exists(target) for target in outputs)
                    else:
                        all_exist = target_exists(outputs)
                    
                    # If all outputs exist, call the callback
                    if all_exist:
                        if callback:
                            callback(outputs)
                        else:
                            # Default callback - can be customized
                            if isinstance(outputs, dict):
                                for key, target in outputs.items():
                                    print(f"Output '{key}' written to {target.path}")
                            elif isinstance(outputs, list):
                                for i, target in enumerate(outputs):
                                    print(f"Output {i} written to {target.path}")
                            else:
                                print(f"Output written to {outputs.path}")
                    
                    return result
                
                # Replace the run method
                self.run = patched_run
                self._run_patched = True
            
            return targets
        
        return wrapper
    
    # Handle both @after_output_written and @after_output_written()
    if func is not None:
        return decorator(func)
    return decorator