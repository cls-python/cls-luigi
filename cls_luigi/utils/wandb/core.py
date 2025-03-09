# core.py
import wandb
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
import numbers
from cls_luigi.utils.wandb._helpers import is_numpy_audio_signal


# Optional dependencies with safe imports

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import torch
except ImportError:
    torch = None

try:
    from tensorflow import keras
except ImportError:
    keras = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    import keras
except ImportError:
    keras = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = None

# Plotly imports
try:
    import plotly.io as pio
    from plotly.graph_objs import Figure
    import plotly.graph_objects as go
except ImportError:
    pio = None
    Figure = None
    go = None

# Module-level state
_initialized = False
_project: Optional[str] = None
_pipeline_name: Optional[str] = None


def wandb_init(
    project_name: str,
    pipeline_name: str,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
    dir: Optional[Union[str, Path]] = None,
    id: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config_exclude_keys: Optional[List[str]] = None,
    config_include_keys: Optional[List[str]] = None,
    allow_val_change: Optional[bool] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    mode: Optional[str] = None,
    force: Optional[bool] = None,
    anonymous: Optional[str] = None,
    reinit: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    resume_from: Optional[str] = None,
    fork_from: Optional[str] = None,
    save_code: Optional[bool] = None,
    tensorboard: Optional[bool] = None,
    sync_tensorboard: Optional[bool] = None,
    monitor_gym: Optional[bool] = None,
    settings: Optional[Union[Dict[str, Any], Any]] = None,
) -> None:
    """Initialize a Weights & Biases run with project and pipeline context.

    This function initializes a new W&B run and sets global variables to track
    the initialization state. It automatically adds the pipeline name to the
    configuration and handles the W&B initialization process.

    Args:
        project_name: Name of the W&B project to log to
        pipeline_name: Name of the current pipeline/run
        config: Optional dictionary of configuration parameters to log
        entity: Optional username or team name where you're sending runs
        dir: Directory where W&B files will be stored
        id: Unique identifier for the run
        notes: Notes about the run to be stored with the run data
        tags: Tags to assign to the run
        config_exclude_keys: Keys to exclude from the config dict
        config_include_keys: Keys to include in the config dict
        allow_val_change: Allow config values to change
        group: Group ID to organize runs together
        job_type: The type of job running
        mode: Can be "online", "offline" or "disabled"
        force: Force a user to be logged in
        anonymous: Can be "never", "allow", or "must"
        reinit: Allow multiple calls to init in the same process
        resume: Resume a previous run
        resume_from: ID of the run to resume from
        fork_from: ID of the run to fork from
        save_code: Save the main script or notebook and create a code artifact
        tensorboard: Sync tensorboard data
        sync_tensorboard: Synchronize wandb logs from tensorboard
        monitor_gym: Monitor Gym environment using videos
        settings: Custom settings for wandb

    Raises:
        RuntimeError: If W&B is already initialized
        ImportError: If wandb package is not installed

    Examples:
        ```python
        # Basic initialization
        wandb_init("my-project", "training-pipeline")

        # With configuration
        wandb_init(
            "my-project",
            "training-pipeline",
            config={"learning_rate": 0.001, "batch_size": 32}
        )

        # With additional W&B parameters
        wandb_init(
            "my-project",
            "training-pipeline",
            tags=["experiment-1", "resnet"],
            notes="Testing improved model architecture",
            group="experiment-group",
            job_type="training"
        )

        # With resume functionality
        wandb_init(
            "my-project",
            "training-pipeline",
            resume="allow",
            id="previous-run-id"
        )
        ```
    """
    global _initialized, _project, _pipeline_name

    # Check for existing wandb run or initialization state
    if wandb.run is not None:
        raise RuntimeError(
            "W&B already initialized. Call wandb_finish() before initializing a new run."
        )

    # Validate inputs
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("project_name must be a non-empty string")
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        raise ValueError("pipeline_name must be a non-empty string")
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dictionary or None")

    # Store global state
    _project = project_name
    _pipeline_name = pipeline_name

    # Prepare configuration with pipeline name
    full_config = {"pipeline_name": pipeline_name}
    if config:
        full_config.update(config)

    # Initialize wandb
    try:
        # Collect all parameters for wandb.init
        init_params = {
            "project": project_name,
            "name": pipeline_name,
            "config": full_config,
            "entity": entity,
            "dir": dir,
            "id": id,
            "notes": notes,
            "tags": tags,
            "config_exclude_keys": config_exclude_keys,
            "config_include_keys": config_include_keys,
            "allow_val_change": allow_val_change,
            "group": group,
            "job_type": job_type,
            "mode": mode,
            "force": force,
            "anonymous": anonymous,
            "reinit": reinit,
            "resume": resume,
            "resume_from": resume_from,
            "fork_from": fork_from,
            "save_code": save_code,
            "tensorboard": tensorboard,
            "sync_tensorboard": sync_tensorboard,
            "monitor_gym": monitor_gym,
            "settings": settings,
        }

        # Remove None values to use wandb defaults
        init_params = {k: v for k, v in init_params.items() if v is not None}

        wandb.init(**init_params)
        _initialized = True
    except Exception as e:
        # Reset global state on failure
        _project = None
        _pipeline_name = None
        raise RuntimeError(f"Failed to initialize W&B: {str(e)}") from e


def wandb_log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """Log data to Weights & Biases with automatic type detection and handling.

    This function provides a flexible interface for logging various types of data to W&B.
    It can handle both direct values and dictionary configurations with special processing.

    Args:
        data: Dictionary mapping names to values or configuration dictionaries.
            - For direct values (non-dictionaries), they are logged directly to W&B.
            - For dictionary values, they are processed based on their inferred or explicit type.
        step: Optional step number for the logged data.
        commit: Whether to commit the data immediately (True) or wait for more data (False).

    Supported data types:
        - metric: Numerical values or dictionaries with a "data" key containing numbers/strings
        - image: W&B Image objects or dictionaries with a "data" key containing:
            - matplotlib figures
            - numpy arrays (2D or 3D)
            - file paths to image files (.png, .jpg, .jpeg)
        - object3d: W&B Object3D objects or dictionaries with a "data" key containing:
            - 3D numpy arrays with shape[2]=3
            - file paths to 3D model files (.obj, .gltf, .glb, .stl, .ply)
        - video: W&B Video objects or file paths to video files
        - audio: W&B Audio objects or file paths to audio files
        - histogram: W&B Histogram objects, lists of numbers, or numpy arrays
        - table: W&B Table objects, pandas DataFrames, or lists of dictionaries
        - html: W&B HTML objects or HTML strings

    Examples:
        ```python
        # Log a simple metric
        wandb_log({"loss": 0.5})

        # Log a metric with a dictionary configuration
        wandb_log({"accuracy": {"data": 0.95, "type": "metric"}})

        # Log an image directly
        wandb_log({"image": wandb.Image("path/to/image.jpg")})

        # Log an image with a dictionary configuration
        wandb_log({"image": {"data": plt.figure(), "type": "image"}})

        # Log a 3D object
        wandb_log({"model": {"data": "path/to/model.obj", "type": "object3d"}})

        # Log multiple items at once
        wandb_log({
            "loss": 0.5,
            "accuracy": 0.95,
            "confusion_matrix": wandb.Image(confusion_matrix_fig),
            "embeddings": {"data": embedding_array, "type": "histogram"}
        })
        ```

    Raises:
        RuntimeError: If W&B is not initialized before calling this function.
        ValueError: If an unsupported data type is provided.
    """

    _check_initialized()

    # Separate direct logging items and items that need processing
    direct_log = {}
    process_log = {}

    for name, data_value in data.items():
        # If data_value is not a dictionary, log it directly
        if not isinstance(data_value, dict):
            direct_log[name] = data_value
        else:
            process_log[name] = data_value

    # Log direct items in one batch if there are any
    if direct_log:
        wandb.log(direct_log, step=step, commit=commit)

    # Process the remaining items that need special handling
    for name, data_dict in process_log.items():
        # Determine the data type, either from the explicit type or by inference
        dtype = (
            data_dict["type"] if "type" in data_dict else _infer_data_type(data_dict)
        )

        try:
            if dtype == "metric":
                _log_metric({name: data_dict}, step=step, commit=commit)

            elif dtype == "image":
                _log_image({name: data_dict}, step=step, commit=commit)

            elif dtype == "video":
                _log_video({name: data_dict}, step=step, commit=commit)

            elif dtype == "audio":
                _log_audio({name: data_dict}, step=step, commit=commit)

            elif dtype == "histogram":
                # For histograms, use the existing _log_histogram function
                _log_histogram({name: data_dict}, step=step, commit=commit)

            elif dtype == "table":
                # For tables, use the existing _log_table function
                _log_table({name: data_dict}, step=step, commit=commit)

            elif dtype == "html":
                # For HTML content, use the existing _log_html function
                _log_html({name: data_dict}, step=step, commit=commit)

            else:
                # raise error not supported
                raise ValueError(f"Data type {dtype} not supported")

        except Exception as e:
            warnings.warn(f"Error logging {name} with type {dtype}: {str(e)}")


def wandb_log_artifact(
    data: Union[Dict[str, Any], Any],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    incremental: bool = False,
    use_as: Optional[str] = None,
) -> None:
    """Log artifacts to Weights & Biases with flexible input formats.

    This function provides a flexible interface for logging artifacts to W&B.
    It supports both the standard W&B artifact API format and a more flexible
    dictionary-based format similar to wandb_log.

    Args:
        data: Either:
            - A single artifact configuration dictionary with required fields
            - A dictionary mapping names to artifact configurations
        name: Name to use when logging a single artifact (ignored if data is a dictionary with multiple items)
        type: Type of artifact to log (e.g., 'model', 'dataset'). Used when logging a single artifact or as a default for artifacts that don't specify a type.
        aliases: Optional list of aliases to apply to the artifact(s)
        tags: Optional list of tags to apply to the artifact(s)
        incremental: If True, only files that have changed since the last version will be saved
        use_as: Optional string to specify how this artifact should be used (e.g., 'model', 'dataset')

    Artifact configuration dictionary format:
        - Standard W&B format (direct artifact API):
        ```python
        {
            'name': str,              # Required: Name of the artifact
            'type': str,              # Required: Type of artifact (e.g., 'model', 'dataset')
            'path': str,              # Required: Path to the file or directory
            'description': str,       # Optional: Description of the artifact
            'metadata': dict,         # Optional: Additional metadata
            'aliases': List[str],     # Optional: Aliases to apply to the artifact
            'tags': List[str],        # Optional: Tags to apply to the artifact
            'incremental': bool,      # Optional: Only save files that have changed since last version
            'use_as': str             # Optional: How this artifact should be used
        }
        ```

        - Flexible format (similar to wandb_log):
        ```python
        {
            'path': str,              # Required: Path to the file or directory
            'type': str,              # Optional: Type of artifact (default: 'file')
            'metadata': dict,         # Optional: Additional metadata
            'description': str,       # Optional: Description of the artifact
            'aliases': List[str],     # Optional: Aliases to apply to the artifact
            'tags': List[str],        # Optional: Tags to apply to the artifact
            'incremental': bool,      # Optional: Only save files that have changed since last version
            'use_as': str             # Optional: How this artifact should be used
        }
        ```

    Examples:
        ```python
        # Standard W&B format
        wandb_log_artifact({
            'name': 'resnet50_model',
            'type': 'model',
            'description': 'ResNet-50 model trained on ImageNet',
            'path': 'models/resnet50.pth',
            'metadata': {'accuracy': 0.76, 'parameters': 25.6e6},
            'aliases': ['best', 'v1'],
            'tags': ['resnet', 'imagenet']
        })

        # Using direct parameters
        wandb_log_artifact(
            {'path': 'models/resnet50.pth'},
            name='resnet50_model',
            type='model',
            aliases=['best', 'v1'],
            tags=['resnet', 'imagenet']
        )

        # Flexible format with a single artifact
        wandb_log_artifact({
            'resnet50_model': {
                'path': 'models/resnet50.pth',
                'type': 'model',
                'metadata': {'accuracy': 0.76},
                'tags': ['resnet', 'imagenet']
            }
        })

        # Multiple artifacts with the flexible format
        wandb_log_artifact({
            'model_weights': {
                'path': 'models/weights.pth',
                'type': 'model',
                'metadata': {'accuracy': 0.76},
                'tags': ['weights', 'trained']
            },
            'training_data': {
                'path': 'data/processed/',
                'type': 'dataset',
                'tags': ['processed']
            }
        })

        # Using with Luigi targets
        wandb_log_artifact({
            'model_output': luigi_target,  # Will use target.path
            'evaluation': {
                'path': evaluation_target.path,
                'type': 'metrics',
                'tags': ['evaluation']
            }
        }, type='output')  # Default type for artifacts that don't specify one
        ```
    """
    _check_initialized()

    # Handle the case where data is a single artifact configuration
    if not isinstance(data, dict) or (
        isinstance(data, dict)
        and "path" in data
        and not any(isinstance(v, dict) for v in data.values())
    ):
        # Single artifact case
        _log_single_artifact(
            data,
            name=name,
            type=type,
            aliases=aliases,
            tags=tags,
            incremental=incremental,
            use_as=use_as,
        )
        return

    # Handle dictionary of artifacts
    for key, value in data.items():
        if isinstance(value, dict) and ("path" in value or hasattr(value, "path")):
            # This is an artifact configuration
            # Pass through parameters if not specified in the config
            artifact_incremental = value.get("incremental", incremental)
            artifact_use_as = value.get("use_as", use_as)
            artifact_aliases = value.get("aliases", aliases)
            artifact_tags = value.get("tags", tags)

            # Get type from value or use the provided type parameter
            artifact_type = value.get("type", type)

            _log_single_artifact(
                value,
                name=key,
                type=artifact_type,
                aliases=artifact_aliases,
                tags=artifact_tags,
                incremental=artifact_incremental,
                use_as=artifact_use_as,
            )
        elif hasattr(value, "path"):
            # This is a Luigi target or similar object with a path attribute
            _log_single_artifact(
                {"path": value.path},
                name=key,
                type=type,
                aliases=aliases,
                tags=tags,
                incremental=incremental,
                use_as=use_as,
            )
        else:
            # Not a recognized artifact format
            warnings.warn(f"Skipping '{key}': not a valid artifact configuration")


def _log_single_artifact(
    artifact_data: Dict[str, Any],
    name: Optional[str] = None,
    type: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    incremental: bool = False,
    use_as: Optional[str] = None,
) -> None:
    """Helper function to log a single artifact to W&B.

    Args:
        artifact_data: Artifact configuration dictionary
        name: Optional name for the artifact (used with the flexible format)
        type: Optional type of artifact (e.g., 'model', 'dataset'). Used as a default if not specified in artifact_data.
        aliases: Optional list of aliases to apply to the artifact
        tags: Optional list of tags to apply to the artifact
        incremental: If True, only files that have changed since the last version will be saved
        use_as: Optional string to specify how this artifact should be used
    """
    # Ensure we have a dictionary or an object with a path attribute
    if not isinstance(artifact_data, dict):
        if hasattr(artifact_data, "path"):
            # Handle Luigi targets or similar objects with a path attribute
            artifact_data = {"path": artifact_data.path}
        else:
            # Neither a dict nor an object with a path attribute
            raise ValueError(
                f"Expected dict or object with 'path' attribute, got {type(artifact_data)}"
            )

    # Ensure we have a path
    if "path" not in artifact_data:
        raise ValueError("Artifact data must contain a 'path' key")

    # Determine if we're using standard W&B format or flexible format
    using_standard_format = "name" in artifact_data and "type" in artifact_data

    # Set up artifact name and type
    if using_standard_format:
        # Standard W&B format
        artifact_name = artifact_data["name"]
        artifact_type = artifact_data["type"]
    else:
        # Flexible format - construct the artifact name if not provided
        if name is None:
            # Use the basename of the path as a fallback name
            name = Path(artifact_data["path"]).stem
        artifact_name = name
        # Use provided type parameter or get from artifact_data, default to 'output'
        artifact_type = artifact_data.get("type", type or "output")

    # Sanitize artifact name - replace path separators with underscores
    # W&B only allows alphanumeric characters, dashes, underscores, and dots in artifact names
    artifact_name = artifact_name.replace("/", "_").replace("\\", "_")

    # Create the artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_data.get("description", ""),
        metadata=artifact_data.get("metadata", {}),
        incremental=artifact_data.get("incremental", incremental),
        use_as=artifact_data.get("use_as", use_as),
    )

    # Get aliases and tags
    artifact_aliases = artifact_data.get("aliases", aliases or [])
    artifact_tags = artifact_data.get("tags", tags or [])

    # Add the file or directory to the artifact
    path = Path(artifact_data["path"])
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    # Log the artifact with aliases and tags
    # wandb.log_artifact supports tags directly as a parameter
    wandb.log_artifact(artifact, aliases=artifact_aliases, tags=artifact_tags)


def wandb_log_model(
    data: Union[Dict[str, Any], str],
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> None:
    """Log machine learning models to Weights & Biases.

    This function provides a flexible interface for logging models to W&B.
    It uses the native wandb.log_model method when possible, and falls back
    to our artifact-based approach for more complex configurations.

    Args:
        data: Either a path to a model file/directory, or a dictionary mapping names to model configurations.
            - For a direct path string, it is logged directly using wandb.log_model.
            - For dictionary values, they are processed based on their configuration.
        name: Optional name for the model artifact when using a direct path string.
            If not provided, W&B will generate a name based on the path.
        aliases: Optional list of aliases to apply to the artifact when using a direct path.

    Configuration dictionary format:
        ```python
        {"model_name": {"path": "path/to/model", ...}}
        ```

        Each model configuration can include:
        ```python
        {
            "path": str,              # Required: Path to the model file or directory
            "name": str,             # Optional: Name for the model (defaults to the key in the data dict)
            "aliases": List[str],    # Optional: Aliases to apply to the artifact
            "tags": List[str],       # Optional: Tags to apply to the artifact
            "metadata": dict,        # Optional: Additional metadata to associate with the model
            "description": str,      # Optional: Description of the model
            "incremental": bool,     # Optional: Whether to save only files that have changed
            "use_as": str           # Optional: How this artifact should be used (typically "model")
        }
        ```

    Examples:
        ```python
        # Log a model directly with a path
        wandb_log_model("models/resnet50.pth", name="resnet_model", aliases=["latest", "best"])

        # Log a model with minimal configuration
        wandb_log_model({"resnet50": {"path": "models/resnet50.pth"}})

        # Log a model with full configuration
        wandb_log_model({
            "bert_model": {
                "path": "models/bert/",
                "aliases": ["best", "v1.0"],
                "tags": ["transformer", "nlp"],
                "metadata": {"accuracy": 0.92, "f1": 0.89, "epochs": 5},
                "description": "Fine-tuned BERT model for text classification",
                "incremental": True,
                "use_as": "model"
            }
        })

        # Log multiple models at once
        wandb_log_model({
            "encoder": {"path": "models/encoder.pt"},
            "decoder": {"path": "models/decoder.pt"}
        })
        ```

    Raises:
        RuntimeError: If W&B is not initialized before calling this function.
        ValueError: If required fields are missing from the configuration.
    """
    _check_initialized()

    # Handle direct path string (use native wandb.log_model)
    if isinstance(data, str):
        wandb.log_model(path=data, name=name, aliases=aliases)
        return

    # Handle dictionary of model configurations
    for name, model_data in data.items():
        try:
            # Simple string path case
            if isinstance(model_data, str):
                wandb.log_model(path=model_data, name=name, aliases=aliases)
                continue

            # Dictionary configuration case
            if isinstance(model_data, dict):
                # Ensure path is provided
                if "path" not in model_data:
                    raise ValueError(f"'path' is required for model artifact: {name}")

                # If it's a simple configuration with just path and maybe aliases,
                # use the native log_model method
                simple_keys = {"path", "aliases"}
                if set(model_data.keys()).issubset(simple_keys):
                    model_aliases = model_data.get("aliases", aliases)
                    wandb.log_model(
                        path=model_data["path"], name=name, aliases=model_aliases
                    )
                    continue

                # For more complex configurations, use our artifact-based approach
                artifact_config = {
                    name: {
                        "path": model_data["path"],
                        "type": "model",
                        "metadata": model_data.get("metadata", {}),
                        "description": model_data.get("description", ""),
                        "aliases": model_data.get("aliases", aliases or ["latest"]),
                        "tags": model_data.get("tags", []),
                        "incremental": model_data.get("incremental", False),
                        "use_as": model_data.get("use_as", "model"),
                    }
                }
                wandb_log_artifact(artifact_config)
            else:
                # For direct model objects (future support)
                warnings.warn(
                    f"Direct model objects not yet supported. Please use a path string or configuration dictionary for {name}"
                )
        except Exception as e:
            warnings.warn(f"Error logging model {name}: {str(e)}")


def wandb_log_plot(
    data: Union[Dict[str, Any], Any],
    name: Optional[str] = None,
    step: Optional[int] = None,
    commit: Optional[bool] = True,
    format: Optional[str] = None,
    dpi: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Safely log plots and visualizations to Weights & Biases.

    This function provides a flexible interface for logging various types of plots to W&B.
    It follows the same pattern as wandb_log, supporting both direct figure objects
    and dictionary configurations with special processing. It handles matplotlib figures,
    plotly figures, and built-in wandb plot types.

    Args:
        data: Either:
            - A single figure object (matplotlib, plotly, or wandb plot) to log with the given name
            - A dictionary mapping names to figure objects or configuration dictionaries
        name: Name to use when logging a single figure (ignored if data is a dictionary)
        step: Optional step number for the logged figure(s)
        commit: Whether to commit immediately
        format: Image format for saving plots ("png", "jpeg", "svg", "pdf")
        dpi: Resolution for rasterized formats
        width: Optional width for the figure in pixels
        height: Optional height for the figure in pixels

    Configuration dictionary format:
        {
            "data": Any,              # Required: The figure object (matplotlib, plotly, vega-lite spec, etc.)
            "type": str,             # Optional: Type identifier (defaults to "plot")
            "format": str,           # Optional: Image format ("png", "jpeg", "svg", "pdf")
            "dpi": int,              # Optional: Resolution for the image
            "width": int,            # Optional: Width in pixels
            "height": int,           # Optional: Height in pixels
            "caption": str,          # Optional: Caption for the plot
            "metadata": dict,        # Optional: Additional metadata to associate with the plot
            "interactive": bool,     # Optional: For Plotly figures, whether to preserve interactivity (default: False)
            "data_table": pd.DataFrame # Optional: For Vega-Lite, the data table to use with the spec
        }

    Plotly Figures:
        For Plotly figures, you can log them as either static images or interactive plots:

        - Static (default): The Plotly figure will be converted to an image
          ```python
          wandb_log_plot({'my_plot': fig})  # fig is a plotly.graph_objects.Figure
          ```

        - Interactive: Set 'interactive': True to preserve interactivity
          ```python
          wandb_log_plot({
              'my_interactive_plot': {
                  'data': fig,
                  'interactive': True,
                  'caption': 'My interactive plot',
                  'metadata': {'key': 'value'}
              }
          })
          ```

        Notes:
        - Interactive Plotly figures are converted to HTML and logged using wandb.Html
        - This ensures all data points are visible and interactive features are preserved
        - Captions are properly handled for both static and interactive plots
        - Metadata is logged separately for interactive plots as '{plot_name}_metadata'
        - For static plots, metadata is attached directly to the image

    Examples:
        ```python
        # Direct logging of a single figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4])
        wandb_log_plot(fig, name="training_loss")

        # Logging with the dictionary format (like wandb_log)
        wandb_log_plot({
            "training_loss": fig,
            "accuracy": px.line(x=[0, 1, 2, 3], y=[0.5, 0.7, 0.8, 0.9])
        })

        # Logging with configuration dictionaries
        wandb_log_plot({
            "training_loss": {
                "data": fig,
                "format": "svg",
                "dpi": 300,
                "caption": "Training loss over time",
                "metadata": {"epoch": 10, "batch_size": 64}
            },
            "interactive_plot": {
                "data": px.line(x=[0, 1, 2, 3], y=[0.5, 0.7, 0.8, 0.9]),
                "interactive": True,  # This preserves the interactive Plotly functionality
                "caption": "Interactive accuracy plot"
            },
            "vega_chart": {
                "data": {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "data": {"values": [{"x": i, "y": i**2} for i in range(10)]},
                    "mark": "line",
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative"},
                        "y": {"field": "y", "type": "quantitative"}
                    }
                },
                "metadata": {"dataset": "synthetic", "model": "quadratic"}
            },
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=[0, 1, 2, 2, 1],
                preds=[0, 1, 2, 1, 0],
                class_names=["class_1", "class_2", "class_3"]
            )
        })

        # Logging built-in wandb plot types
        wandb_log_plot({
            "pr_curve": wandb.plot.pr_curve(
                y_true=[0, 0, 1, 1],
                y_probas=[0.1, 0.4, 0.35, 0.8],
                labels=["class_0", "class_1"]
            )
        })
        ```

        ```python
        # Logging wandb.plot.line with metadata and caption
        import pandas as pd
        import numpy as np

        # Create a table for the line plot
        data = np.random.rand(10)
        x_values = list(range(len(data)))
        table = pd.DataFrame({"x": x_values, "y": data})

        # Create the plot dictionary with metadata and caption
        wandb_log_plot({
            "line_plot_with_metadata": {
                "data": wandb.plot.line(
                    table,
                    "x",
                    "y",
                    title="Example Line Plot"
                ),
                "caption": "Line plot showing random data over time",
                "metadata": {
                    "data_points": len(data),
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "experiment_id": "exp_001"
                }
            }
        })
        ```

        ```python
        # Logging an interactive Plotly Express scatter plot with the Iris dataset
        import plotly.express as px

        # Create a scatter plot with the Iris dataset
        df = px.data.iris()
        fig = px.scatter(
            df,
            x="sepal_width",
            y="sepal_length",
            color="species",
            size="petal_length",
            hover_data=["petal_width"]
        )

        # Customize layout
        fig.update_layout(
            title='Iris Dataset Scatter Plot',
            xaxis_title='Sepal Width',
            yaxis_title='Sepal Length',
            template='plotly_white'
        )

        # Log the plot with interactive mode and metadata
        wandb_log_plot({
            'iris_scatter': {
                'data': fig,
                'interactive': True,  # Enable interactive features
                'caption': 'Interactive Iris dataset scatter plot showing sepal width vs length',
                'metadata': {
                    'dataset': 'iris',
                    'features': ['sepal_width', 'sepal_length', 'species', 'petal_length', 'petal_width'],
                    'num_samples': len(df)
                }
            }
        })
        ```
    """
    _check_initialized()

    # Handle the case where data is a single figure object
    if not isinstance(data, dict):
        if name is None:
            name = "plot"  # Default name if not provided

        # Create a dictionary with the single figure
        plot_data = {name: data}
    else:
        # Use the provided dictionary
        plot_data = data

    # Separate direct logging items and items that need processing
    direct_log = {}
    processed_images = {}

    for plot_name, plot_value in plot_data.items():
        try:
            # Handle wandb built-in plot types directly
            if isinstance(plot_value, wandb.plot.CustomChart):
                direct_log[plot_name] = plot_value
                continue

            # Process configuration dictionaries
            if isinstance(plot_value, dict) and "data" in plot_value:
                # Extract configuration options
                figure = plot_value["data"]
                plot_format = plot_value.get("format", format)
                plot_dpi = plot_value.get("dpi", dpi)
                plot_width = plot_value.get("width", width)
                plot_height = plot_value.get("height", height)
                caption = plot_value.get("caption", None)
                metadata = plot_value.get("metadata", {})

                # Handle wandb built-in plot types inside the dictionary
                if isinstance(figure, wandb.plot.CustomChart):
                    # Create a new dictionary with the plot and metadata/caption
                    if caption or metadata:
                        plot_dict = {"plot": figure}
                        if caption:
                            plot_dict["caption"] = caption
                        if metadata:
                            plot_dict["metadata"] = metadata
                        direct_log[plot_name] = plot_dict
                    else:
                        direct_log[plot_name] = figure
                    continue
            else:
                # Use the figure directly with default options
                figure = plot_value
                plot_format = format  # Will be None if not specified
                plot_dpi = dpi
                plot_width = width
                plot_height = height
                caption = None
                metadata = {}

            # Process different figure types

            # Handle Matplotlib figures
            if plt and isinstance(figure, plt.Figure):
                # Save the original DPI to restore it later
                original_dpi = figure.dpi

                # Set the DPI if explicitly specified
                if plot_dpi is not None:
                    figure.dpi = plot_dpi

                # Create a wandb.Image with the figure
                if caption or metadata:
                    processed_images[plot_name] = wandb.Image(
                        figure, caption=caption, metadata=metadata
                    )
                else:
                    processed_images[plot_name] = figure

                # Restore original DPI
                figure.dpi = original_dpi

                # Close the figure to prevent memory leaks
                plt.close(figure)

            # Handle Plotly figures
            elif pio and (
                (Figure and isinstance(figure, Figure))
                or (isinstance(figure, dict) and "data" in figure)
                or (hasattr(figure, "to_plotly_json"))
            ):
                # Determine if we should preserve interactivity
                preserve_interactive = (
                    plot_value.get("interactive", False)
                    if isinstance(plot_value, dict)
                    else False
                )

                # First, ensure we have a proper Plotly Figure object
                try:
                    # Convert various inputs to a standard Plotly Figure
                    if hasattr(figure, "to_plotly_json"):
                        # Already a Plotly Figure object
                        plotly_figure = figure
                    elif (
                        isinstance(figure, dict)
                        and "data" in figure
                        and "layout" in figure
                    ):
                        # JSON-like dictionary representation
                        plotly_figure = go.Figure(figure)
                    else:
                        # Try other conversions
                        if hasattr(figure, "data") and hasattr(figure, "layout"):
                            # Object with data and layout attributes
                            plotly_figure = go.Figure(
                                data=figure.data, layout=figure.layout
                            )
                        else:
                            # Last resort, try direct conversion
                            plotly_figure = figure
                            if not hasattr(plotly_figure, "to_plotly_json"):
                                # If we still don't have a proper Figure, raise an error
                                raise ValueError(
                                    f"Could not convert to Plotly Figure: {type(figure)}"
                                )

                    # Now handle based on whether we want interactive or static
                    if preserve_interactive:
                        # For interactive plots, use HTML representation which preserves all data points
                        # Convert to HTML string
                        html_string = pio.to_html(plotly_figure, include_plotlyjs="cdn")

                        # Create W&B HTML object
                        html_obj = wandb.Html(html_string)

                        # Add to direct log with proper caption
                        if caption:
                            # We can't set caption directly on Html objects, so include it in the log dict
                            direct_log[plot_name] = {
                                "plot": html_obj,
                                "caption": caption,
                            }
                        else:
                            direct_log[plot_name] = html_obj

                        # Log metadata separately if provided
                        if metadata:
                            direct_log[f"{plot_name}_metadata"] = metadata
                    else:
                        # For static plots, convert to image
                        to_image_kwargs = {}
                        if plot_format is not None:
                            to_image_kwargs["format"] = plot_format
                        if plot_width is not None:
                            to_image_kwargs["width"] = plot_width
                        if plot_height is not None:
                            to_image_kwargs["height"] = plot_height

                        # Convert to image bytes
                        img_bytes = pio.to_image(plotly_figure, **to_image_kwargs)

                        # Create wandb.Image with caption and metadata
                        if caption or metadata:
                            img_obj = wandb.Image(
                                img_bytes, caption=caption, metadata=metadata
                            )
                        else:
                            img_obj = wandb.Image(img_bytes)

                        # Add to processed images
                        processed_images[plot_name] = img_obj

                except Exception as e:
                    warnings.warn(f"Error processing Plotly figure: {str(e)}")
                    # Try fallback to static image if we have a figure but couldn't process it properly
                    try:
                        if hasattr(figure, "to_plotly_json") or (
                            isinstance(figure, dict) and "data" in figure
                        ):
                            img_bytes = pio.to_image(figure)
                            img_obj = wandb.Image(img_bytes, caption=caption)
                            processed_images[plot_name] = img_obj
                            if metadata:
                                direct_log[f"{plot_name}_metadata"] = metadata
                    except Exception as fallback_error:
                        warnings.warn(
                            f"Fallback to static image also failed: {str(fallback_error)}"
                        )

            # Handle JSON strings for Plotly
            elif pio and isinstance(figure, str):
                try:
                    # Attempt to parse as JSON
                    figure_dict = pio.from_json(figure)

                    # Check if we should preserve interactivity
                    preserve_interactive = (
                        plot_value.get("interactive", False)
                        if isinstance(plot_value, dict)
                        else False
                    )

                    # Handle interactive Plotly figures from JSON
                    if preserve_interactive:
                        # Create a Plotly object for interactive visualization
                        try:
                            # Convert the JSON dict to a proper Plotly Figure object
                            plotly_figure = go.Figure(figure_dict)

                            # Create the Plotly object for W&B
                            plotly_obj = wandb.Plotly(plotly_figure)

                            # Set caption if provided
                            if caption:
                                plotly_obj._caption = caption

                            # Add to direct log
                            direct_log[plot_name] = plotly_obj

                            # Log metadata separately if provided
                            if metadata:
                                direct_log[f"{plot_name}_metadata"] = metadata
                        except Exception as e:
                            warnings.warn(
                                f"Error creating interactive Plotly from JSON: {str(e)}"
                            )
                            # Fallback to static image
                            preserve_interactive = False

                    # Handle static Plotly figures from JSON
                    if not preserve_interactive:
                        # Convert to static image
                        to_image_kwargs = {}
                        if plot_format is not None:
                            to_image_kwargs["format"] = plot_format
                        if plot_width is not None:
                            to_image_kwargs["width"] = plot_width
                        if plot_height is not None:
                            to_image_kwargs["height"] = plot_height

                        try:
                            # Convert to image bytes
                            img_bytes = pio.to_image(figure_dict, **to_image_kwargs)

                            # Create wandb.Image with caption and metadata
                            img_obj = wandb.Image(img_bytes)
                            if caption:
                                img_obj.caption = caption

                            # Add to processed images
                            processed_images[plot_name] = img_obj

                            # Log metadata separately if provided
                            if metadata:
                                direct_log[f"{plot_name}_metadata"] = metadata
                        except Exception as e:
                            warnings.warn(
                                f"Error converting Plotly JSON to image: {str(e)}"
                            )
                except Exception as e:
                    raise ValueError("Invalid JSON string for Plotly figure.") from e

            # Handle numpy arrays as images
            elif isinstance(figure, np.ndarray):
                if caption or metadata:
                    processed_images[plot_name] = wandb.Image(
                        figure, caption=caption, metadata=metadata
                    )
                else:
                    processed_images[plot_name] = wandb.Image(figure)

            # Handle file paths
            elif isinstance(figure, (str, Path)) and Path(figure).is_file():
                if caption or metadata:
                    processed_images[plot_name] = wandb.Image(
                        str(figure), caption=caption, metadata=metadata
                    )
                else:
                    processed_images[plot_name] = wandb.Image(str(figure))

            # Handle Vega-Lite specifications
            elif isinstance(figure, dict) and (
                "$schema" in figure
                or figure.get("type") == "vega-lite"
                or "mark" in figure
                or "layer" in figure
                or "hconcat" in figure
                or "vconcat" in figure
            ):
                # Extract data table if provided
                data_table = (
                    plot_value.get("data_table", None)
                    if isinstance(plot_value, dict)
                    else None
                )

                # Log the Vega-Lite chart
                if metadata:
                    direct_log[plot_name] = wandb.plots.VegaLite(
                        figure, data_table=data_table, metadata=metadata
                    )
                else:
                    direct_log[plot_name] = wandb.plots.VegaLite(
                        figure, data_table=data_table
                    )

            else:
                raise ValueError(
                    f"Unsupported figure type for {plot_name}: {type(figure)}. "
                    "Supported types include matplotlib.figure.Figure, plotly.graph_objs.Figure, "
                    "numpy arrays, file paths, Vega-Lite specifications, or wandb plot objects."
                )

        except Exception as e:
            warnings.warn(f"Error processing plot {plot_name}: {str(e)}")

    # Log direct items if there are any
    if direct_log:
        wandb.log(direct_log, step=step, commit=commit)

    # Log processed images if there are any
    if processed_images:
        wandb.log(processed_images, step=step, commit=commit)
    # If nothing was logged but commit is True, log an empty dict to trigger the commit
    elif commit and not direct_log:
        wandb.log({}, step=step, commit=commit)


def wandb_finish() -> None:
    """Finalize the current Weights & Biases run.

    This function safely finalizes the current W&B run if one exists and resets
    the global initialization state. It should be called at the end of your
    pipeline or experiment to properly close the W&B session.

    Note that this function does nothing if W&B has not been initialized.

    Examples:
        ```python
        # After initializing and logging data
        wandb_init("my-project", "training-pipeline")

        # ... perform logging operations ...
        wandb_log({"loss": 0.5, "accuracy": 0.95})

        # Log a model artifact
        wandb_log_model("path/to/model.pth", name="final_model")

        # Properly close the W&B run
        wandb_finish()
        ```
    """
    global _initialized, _project, _pipeline_name

    if _initialized:
        try:
            # Attempt to finish the wandb run
            wandb.finish()
        except Exception as e:
            warnings.warn(f"Error while finishing W&B run: {str(e)}")
        finally:
            # Reset global state even if there was an error
            _initialized = False
            _project = None
            _pipeline_name = None


def wandb_get_status() -> Dict[str, Any]:
    """Get the current status of the Weights & Biases run.

    This function returns information about the current W&B run, including
    whether it's initialized, the project name, pipeline name, and the run URL.

    Returns:
        Dict[str, Any]: Dictionary containing status information with the following keys:
            - 'initialized': Boolean indicating if W&B is initialized
            - 'project': Name of the current project (if initialized)
            - 'pipeline_name': Name of the current pipeline (if initialized)
            - 'run_url': URL to the W&B run (if initialized)

    Examples:
        >>> status = wandb_get_status()
        >>> print(f"W&B initialized: {status['initialized']}")
        >>> if status['initialized']:
        ...     print(f"Project: {status['project']}")
        ...     print(f"Pipeline: {status['pipeline_name']}")
        ...     print(f"Run URL: {status['run_url']}")
    """
    """Return current W&B configuration status"""
    return {
        "initialized": _initialized,
        "project": _project,
        "pipeline": _pipeline_name,
        "dependencies": {
            "numpy": np is not None,
            "pandas": pd is not None,
            "matplotlib": plt is not None,
            "pytorch": torch is not None,
            "tensorflow": keras is not None,
            "joblib": joblib is not None,
            "xgboost": xgb is not None,
            "lightgbm": lgb is not None,
            "scikit-learn": BaseEstimator is not None,
            "plotly": pio is not None,
        },
    }


# Internal helper functions


def _log_metric(
    metric: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """Log metrics to Weights & Biases.

    This function logs scalar metrics to W&B. It handles both simple scalar values
    and metrics wrapped in a dictionary with a 'data' key.

    Args:
        metric: Dictionary mapping metric names to values, which can be:
            - Scalar values (int, float)
            - Dictionaries with a 'data' key containing scalar values
        step: Optional step number for the logged metrics
        commit: Whether to commit the metrics immediately

    Examples:
        # Simple scalar metrics
        >>> _log_metric({"accuracy": 0.95, "loss": 0.05})

        # With step number
        >>> _log_metric({"accuracy": 0.97, "loss": 0.03}, step=100)

        # Using data dictionary format
        >>> _log_metric({
        ...     "validation_metrics": {
        ...         "data": {"precision": 0.92, "recall": 0.89, "f1": 0.90}
        ...     }
        ... })
    """
    for name, metric_data in metric.items():
        if "metadata" in metric_data and "data" in metric_data:
            wandb.log(
                {name: metric_data["data"], **metric_data["metadata"]},
                step=step,
                commit=commit,
            )
        elif "data" in metric_data:
            wandb.log({name: metric_data["data"]}, step=step, commit=commit)
        else:
            warnings.warn(
                f"Unsupported metric data type for key {name}: {type(metric_data)}. Skipping."
            )


def _log_histogram(
    histogram_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """Log histogram data to Weights & Biases.

    This function handles different types of histogram data and converts them to
    wandb.Histogram objects for logging. It supports lists of values, NumPy arrays,
    and pre-computed histograms.

    Args:
        histogram_data: Dictionary mapping names to histogram data, which can be:
            - wandb.Histogram objects
            - Dictionaries with a "data" key containing:
                - Lists of numeric values
                - NumPy arrays with numeric values
                - Tuples of NumPy arrays (as returned by np.histogram)
        step: Optional step number for the logged histograms
        commit: Whether to commit the histograms immediately

    Examples:
        # With a list of values
        >>> _log_histogram({"scores": {"data": [65, 72, 84, 91, 95]}})

        # With a numpy array
        >>> import numpy as np
        >>> _log_histogram({"weights": {"data": np.random.normal(0, 1, 1000)}})

        # With a pre-computed histogram
        >>> hist, bin_edges = np.histogram(data, bins=20)
        >>> _log_histogram({"distribution": {"data": (hist, bin_edges)}})
    """

    try:
        processed = {}
        for key, value in histogram_data.items():
            if isinstance(value, wandb.Histogram):
                processed[key] = value
                break

            # If value is a dictionary with a "data" key, process the inner value
            if isinstance(value, dict) and "data" in value:
                inner_value = value["data"]
                histogram_kwargs = {k: v for k, v in value.items() if k != "data"}

                if isinstance(inner_value, list) and all(
                    isinstance(item, (int, float)) for item in inner_value
                ):
                    # Create a wandb Histogram from the list of numeric values
                    processed[key] = wandb.Histogram(inner_value)
                elif np is not None and isinstance(
                    inner_value, Tuple[np.ndarray, np.ndarray]
                ):
                    # Create a wandb Histogram from the NumPy array
                    processed[key] = wandb.Histogram(np_histogram=inner_value)
                elif (
                    np is not None
                    and isinstance(inner_value, np.ndarray)
                    and np.issubdtype(inner_value.dtype, np.number)
                ):
                    # Create a wandb Histogram from the NumPy array
                    processed[key] = wandb.Histogram(inner_value)
                else:
                    raise ValueError(
                        f"Unsupported histogram format for key {key}: {type(inner_value)}. Skipping."
                    )

        # Log the processed histograms
        if processed:
            wandb.log(processed, step=step, commit=commit)
    except Exception as e:
        warnings.warn(f"Error logging histogram data: {str(e)}")


def _log_table(
    table: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """Log tabular data to Weights & Biases.

    This function handles various formats of tabular data and converts them to
    wandb.Table objects for logging. It supports pandas DataFrames, dictionaries,
    lists of lists, lists of dictionaries, and numpy arrays.

    Args:
        table: Dictionary mapping names to table data, which can be:
            - wandb.Table objects
            - pandas DataFrames
            - Dictionaries with lists as values
            - Lists of lists with a 'columns' key
            - Lists of dictionaries
            - NumPy arrays with a 'columns' key
            - Dictionaries with pandas Series as values
        step: Optional step number for the logged tables
        commit: Whether to commit the tables immediately

    Examples:
        # With a pandas DataFrame
        >>> import pandas as pd
        >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        >>> _log_table({"my_table": df})

        # With a list of dictionaries
        >>> data = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        >>> _log_table({"points": data})

        # With a list of lists and columns
        >>> data = {"matrix": {"data": [[1, 2], [3, 4]], "columns": ["a", "b"]}}
        >>> _log_table(data)

        # With a numpy array and columns
        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> _log_table({"array_table": {"data": arr, "columns": ["col1", "col2"]}})
    """
    try:
        processed = {}
        for key, value in table.items():
            if isinstance(value, wandb.Table):
                processed[key] = value
                break

            # If value is a dictionary with a "data" key, process the inner value
            if isinstance(value, dict) and "data" in value:
                inner_value = value["data"]
                table_kwargs = {k: v for k, v in value.items() if k != "data"}

                if pd and isinstance(inner_value, pd.DataFrame):
                    processed[key] = wandb.Table(dataframe=inner_value, **table_kwargs)
                elif (
                    pd
                    and isinstance(inner_value, dict)
                    and isinstance(inner_value.keys(), str)
                    and all(isinstance(k, list) for k in inner_value.values())
                ):
                    processed[key] = wandb.Table(
                        dataframe=pd.DataFrame(inner_value), **table_kwargs
                    )
                elif (
                    isinstance(inner_value, list)
                    and all(isinstance(item, list) for item in inner_value)
                    and "columns" in value
                    and all(isinstance(col, str) for col in value["columns"])
                ):
                    processed[key] = wandb.Table(
                        columns=value["columns"], data=inner_value, **table_kwargs
                    )
                elif isinstance(inner_value, list) and all(
                    isinstance(item, dict) for item in inner_value
                ):
                    processed[key] = wandb.Table(data=inner_value, **table_kwargs)
                elif (
                    np is not None
                    and isinstance(inner_value, np.ndarray)
                    and "columns" in value
                    and all(isinstance(col, str) for col in value["columns"])
                ):
                    processed[key] = wandb.Table(
                        data=inner_value, columns=value["columns"], **table_kwargs
                    )
                elif (
                    pd
                    and isinstance(inner_value, dict)
                    and isinstance(inner_value.keys(), str)
                    and all(isinstance(k, pd.Series) for k in inner_value.values())
                ):
                    processed[key] = wandb.Table(
                        dataframe=pd.DataFrame(inner_value), **table_kwargs
                    )
                else:
                    raise ValueError(
                        f"Unsupported table format for key {key}: {type(inner_value)}. Skipping."
                    )

        if processed:
            wandb.log(processed, step=step, commit=commit)

    except Exception as e:
        warnings.warn(f"Error logging table data: {str(e)}")


def _log_image(
    image_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """Log image data to Weights & Biases.

    This function handles various image formats and converts them to a format
    that can be logged to W&B. It supports PIL images, NumPy arrays, file paths,
    and 3D objects.

    Args:
        image_data: Dictionary mapping names to image data, which can be:
            - wandb.Image objects
            - PIL Image objects
            - NumPy arrays (2D for grayscale, 3D for RGB/RGBA)
            - File paths to image files
            - Dictionaries with a 'data' key containing any of the above
        step: Optional step number for the logged images
        commit: Whether to commit the images immediately

    Examples:
        # With a PIL image
        >>> from PIL import Image
        >>> img = Image.open('sample.jpg')
        >>> _log_image({"sample_image": img})

        # With a numpy array
        >>> import numpy as np
        >>> array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> _log_image({"random_image": array})

        # With a file path
        >>> _log_image({"external_image": {"data": "path/to/image.png"}})

        # With a 3D object (for 3D visualization)
        >>> vertices = np.random.rand(100, 3)
        >>> _log_image({"point_cloud": {"data": vertices, "type": "3d"}})

    """
    try:
        processed = {}

        for key, value in image_data.items():
            # If value is already a wandb.Image or wandb.Object3D, use it directly
            if isinstance(value, (wandb.Image, wandb.Object3D)):
                processed[key] = value
                break

            # If value is a dictionary with a "data" key, process the inner value
            if isinstance(value, dict) and "data" in value:
                inner_value = value["data"]

                # Extract optional caption and metadata if present
                caption = value.get("caption")
                metadata = value.get("metadata")
                # Create kwargs dictionary with only the parameters that are provided
                img_kwargs = {}
                if caption is not None:
                    img_kwargs["caption"] = caption
                if metadata is not None:
                    img_kwargs["metadata"] = metadata

                # Handle matplotlib figure
                if plt and isinstance(inner_value, plt.Figure):
                    # Create the wandb.Image with the appropriate parameters
                    if img_kwargs:
                        processed[key] = wandb.Image(inner_value, **img_kwargs)
                    else:
                        processed[key] = wandb.Image(inner_value)
                    plt.close(inner_value)  # Close the figure to prevent memory leaks
                    break

                # Handle numpy array
                elif np is not None and isinstance(inner_value, np.ndarray):
                    if inner_value.ndim in [2, 3]:
                        # Check if it's a 3D object (3D array with 3 channels for RGB)
                        if inner_value.ndim == 3 and inner_value.shape[2] == 3:
                            if img_kwargs:
                                processed[key] = wandb.Object3D(
                                    inner_value, **img_kwargs
                                )
                            else:
                                processed[key] = wandb.Object3D(inner_value)

                        else:
                            if img_kwargs:
                                processed[key] = wandb.Image(inner_value, **img_kwargs)
                            else:
                                processed[key] = wandb.Image(inner_value)
                    break

                # Handle file path
                elif isinstance(inner_value, (str, Path)):
                    path = Path(inner_value)
                    if path.exists():
                        suffix = path.suffix.lower()
                        if suffix in [".png", ".jpg", ".jpeg"]:
                            if img_kwargs:
                                processed[key] = wandb.Image(str(path), **img_kwargs)
                            else:
                                processed[key] = wandb.Image(str(path))
                        # Add support for 3D file formats
                        elif suffix in [".obj", ".gltf", ".glb", ".stl", ".ply"]:
                            if img_kwargs:
                                processed[key] = wandb.Object3D(str(path), **img_kwargs)
                            else:
                                processed[key] = wandb.Object3D(str(path))
                        break
                else:
                    raise ValueError(
                        f"Unsupported image format for key {key}. Skipping."
                    )
            else:
                raise ValueError(f"Unsupported image format for key {key}. Skipping.")

        if processed:
            wandb.log(processed, step=step, commit=commit)
    except Exception as e:
        warnings.warn(f"Error logging image data: {str(e)}")


def _log_video(
    video_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """
    Log video data to wandb

    Args:
        video_data: Dictionary mapping names to video data
        step: Optional step for logging
        commit: Whether to commit the log immediately

    Examples:
        _log_video({"video": wandb.Video("path/to/video.mp4")})
        _log_video({"video": {"data": "path/to/video.mp4", "fps": 30, "caption": "My video"}})
    """
    try:
        processed = {}

        for key, value in video_data.items():
            # If value is already a wandb.Video, use it directly
            if isinstance(value, wandb.Video):
                processed[key] = value
                break

            # If value is a dictionary with a "data" key, process the inner value
            if isinstance(value, dict) and "data" in value:
                inner_value = value["data"]
                video_kwargs = {k: v for k, v in value.items() if k != "data"}
                # If inner_value is already a wandb.Video, use it directly
                if isinstance(inner_value, wandb.Video):
                    processed[key] = inner_value
                    break

                # handle numpy array as video
                if isinstance(inner_value, np.ndarray):
                    processed[key] = wandb.Video(inner_value, **video_kwargs)
                    break

                # Handle file path
                if isinstance(inner_value, (str, Path)):
                    path = Path(inner_value)
                    if path.exists() and path.suffix.lower() in [".mp4", ".gif"]:
                        # Extract video-specific parameters
                        processed[key] = wandb.Video(str(path), **video_kwargs)
                        break
                    else:
                        raise ValueError(
                            f"Unsupported or non-existent video file for key {key}: {path}. Skipping."
                        )
                else:
                    raise ValueError(
                        f"Unsupported video data type for key {key}: {type(inner_value)}. Skipping."
                    )
            else:
                raise ValueError(
                    f"Missing 'data' key in video configuration for key {key}. Skipping."
                )

        if processed:
            wandb.log(processed, step=step, commit=commit)
    except Exception as e:
        warnings.warn(f"Error logging video data: {str(e)}")


def _log_audio(
    audio_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    """Log audio data to Weights & Biases.

    This function handles various audio formats and converts them to a format
    that can be logged to W&B. It supports numpy arrays with audio signals,
    file paths to audio files, and wandb.Audio objects.

    Args:
        audio_data: Dictionary mapping names to audio data, which can be:
            - wandb.Audio objects
            - NumPy arrays containing audio signals
            - File paths to audio files (.wav, .mp3)
            - Dictionaries with a 'data' key containing any of the above
        step: Optional step number for the logged audio
        commit: Whether to commit the audio immediately

    Examples:
        ```python
        # With a file path
        _log_audio({"sample_audio": {"data": "path/to/audio.wav"}})

        # With a numpy array (assuming sample_rate=16000)
        import numpy as np
        audio_array = np.random.uniform(-1, 1, 16000)
        _log_audio({"generated_audio": {"data": audio_array, "sample_rate": 16000}})

        # With a wandb.Audio object
        _log_audio({"voice_sample": wandb.Audio("recording.mp3", caption="Voice recording")})
        ```
    """
    try:
        processed = {}
        for key, value in audio_data.items():
            if isinstance(value, wandb.Audio):
                processed[key] = value
                break

            if isinstance(value, dict) and "data" in value:
                inner_value = value["data"]
                audio_kwargs = {k: v for k, v in value.items() if k != "data"}

                # handle numpy array as audio
                if np is not None and is_numpy_audio_signal(inner_value):
                    processed[key] = wandb.Audio(inner_value, **audio_kwargs)
                    break

                if isinstance(inner_value, (str, Path)):
                    path = Path(inner_value)
                    if path.exists() and path.suffix.lower() in [".wav", ".mp3"]:
                        processed[key] = wandb.Audio(str(path), **audio_kwargs)
                        break
                else:
                    raise ValueError(
                        f"Unsupported or non-existent audio file for key {key}: {path}. Skipping."
                    )
            else:
                raise ValueError(
                    f"Unsupported audio data type for key {key}: {type(value)}. Skipping."
                )

        if processed:
            wandb.log(processed, step=step, commit=commit)

    except Exception as e:
        warnings.warn(f"Error logging audio data: {str(e)}")


def _log_html(
    html_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: Optional[bool] = True,
) -> None:
    try:
        processed = {}
        for key, value in html_data.items():
            if isinstance(value, str):
                processed[key] = wandb.Html(value)
                break

            if isinstance(value, dict) and "html" in value:
                inner_value = value["html"]
                html_kwargs = {k: v for k, v in value.items() if k != "html"}

                if isinstance(inner_value, str):
                    processed[key] = wandb.Html(inner_value, **html_kwargs)
                else:
                    raise ValueError(
                        f"Unsupported or non-existent HTML file for key {key}: {inner_value}. Skipping."
                    )

        if processed:
            wandb.log(processed, step=step, commit=commit)
    except Exception as e:
        warnings.warn(f"Error logging HTML data: {str(e)}")


def _detect_framework(model: Any) -> str:
    """Auto-detect ML framework with safety checks"""

    # Check for PyTorch
    if torch and isinstance(model, torch.nn.Module):
        return "pytorch"

    # Check for TensorFlow/Keras
    if keras and isinstance(model, keras.Model):
        return "tensorflow"

    # Check for Scikit-Learn
    if BaseEstimator and isinstance(model, BaseEstimator):
        return "scikit-learn"

    # Check for XGBoost
    if xgb and isinstance(model, xgb.Booster):
        return "xgboost"

    # Check for LightGBM
    if lgb and isinstance(model, lgb.Booster):
        return "lightgbm"

    # Check for Joblib
    if joblib and isinstance(model, joblib.Parallel):
        return "joblib"

    raise RuntimeError(
        "Could not detect framework - install PyTorch/TensorFlow/Joblib/XGBoost/LightGBM/Scikit-Learn"
    )


def _infer_data_type(data: Dict[str, Any]) -> str:
    """Infer the type of data to be logged to Weights & Biases.

    This function examines the provided data dictionary and determines the most
    appropriate logging method based on the content types. It can detect various
    data types including metrics, histograms, images, videos, audio, HTML, and tables.

    Args:
        data: Dictionary of data to be logged

    Returns:
        str: The inferred data type, one of: 'metric', 'histogram', 'image',
             'video', 'audio', 'html', 'table', or 'unknown'

    Examples:
        >>> _infer_data_type({"loss": 0.5, "accuracy": 0.95})
        'metric'

        >>> import numpy as np
        >>> _infer_data_type({"weights": np.random.normal(0, 1, 1000)})
        'histogram'

        >>> from PIL import Image
        >>> _infer_data_type({"sample": Image.new('RGB', (100, 100))})
        'image'
    """
    # Check for wandb specific objects
    if any(isinstance(v, wandb.Image) for v in data.values()):
        return "image"

    # Check for image data in the "data" key
    if "data" in data:
        value = data["data"]
        # Check if it's a matplotlib figure
        if plt and isinstance(value, plt.Figure):
            return "image"
        # Check if it's a numpy array that could be an image
        elif np is not None and isinstance(value, np.ndarray):
            if value.ndim in [2, 3]:  # 2D grayscale or 3D color image
                return "image"
        # Check if it's a file path to an image
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if path.exists() and path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                return "image"

    if any(isinstance(v, wandb.Video) for v in data.values()):
        return "video"

    if "data" in data:
        value = data["data"]
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.exists() and path.suffix.lower() in [".mp4", ".gif"]:
                return "video"

    if any(isinstance(v, wandb.Audio) for v in data.values()):
        return "audio"

    if "data" in data:
        value = data["data"]

        if np is not None and is_numpy_audio_signal(value):
            return "audio"
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if path.exists() and path.suffix.lower() in [".wav", ".mp3"]:
                return "audio"

    if any(isinstance(v, wandb.Html) for v in data.values()):
        return "html"

    if "html" in data:
        value = data["html"]
        if isinstance(value, str):
            return "html"

    if any(isinstance(v, wandb.Table) for v in data.values()):
        return "table"

    if any(isinstance(v, wandb.Histogram) for v in data.values()):
        return "histogram"

    # Check for pandas DataFrame (if pandas is available)
    if pd and any(isinstance(v, pd.DataFrame) for v in data.values()):
        return "table"

    if np is not None and any(
        isinstance(v, Tuple[np.ndarray, np.ndarray]) for v in data.values()
    ):
        return "histogram"

    # Check for histogram data (list of numbers or numpy array)
    if any(
        (isinstance(v, list) and all(isinstance(item, (int, float)) for item in v))
        or (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number))
        for v in data.values()
    ):
        return "histogram"

    # Check for table data (dict format for Dataframe)
    if any(
        isinstance(v, dict)
        and isinstance(v.keys(), str)
        and all(isinstance(k, list) for k in v.values())
        for v in data.values()
    ):
        return "table"

    # Check for table data (list of dictionaries) => in DataFrames format
    if any(
        isinstance(v, list) and all(isinstance(item, dict) for item in v)
        for v in data.values()
    ):
        return "table"

    # Check for table data (list of lists)
    if any(
        isinstance(v, list)
        and all(isinstance(row, list) for row in v)
        and "columns" in data
        and all(isinstance(col, str) for col in data["columns"])
        for _, v in data.items()
    ):
        return "table"

    # Check for table data (np.ndarray with "columns" key)
    if any(
        isinstance(v, np.ndarray)
        and "columns" in data
        and all(isinstance(col, str) for col in data["columns"])
        for _, v in data.items()
    ):
        return "table"

    # Check for table data (dict with pd.Series)
    if any(
        isinstance(v, dict)
        and isinstance(v.keys(), str)
        and all(isinstance(k, pd.Series) for k in v.values())
        for v in data.values()
    ):
        return "table"

    # Check for table data (dict format for Dataframe)
    if any(
        isinstance(v, dict)
        and isinstance(v.keys(), str)
        and all(isinstance(k, list) for k in v.values())
        for v in data.values()
    ):
        return "table"

    # Default to metric for any other type
    return (
        "metric"
        if "data" in data.values()
        and (
            isinstance(data["data"], (numbers.Number, str))
            or (np and isinstance(data["data"], np.number))
        )
        else "none"
    )


def _check_initialized() -> None:
    """Check if Weights & Biases has been initialized.

    This utility function verifies that W&B has been properly initialized
    before attempting to log data. It's used internally by the logging
    functions to ensure proper usage order.

    Raises:
        RuntimeError: If W&B has not been initialized with wandb_init()
    """
    global _initialized

    # First check if wandb.run exists (in case wandb was initialized outside this module)
    if wandb.run is None and not _initialized:
        raise RuntimeError(
            "W&B not initialized. Call wandb_init() first before attempting to log data."
        )

    # If we get here but our internal flag is not set, update it to match reality
    if wandb.run is not None and not _initialized:
        global _project, _pipeline_name
        _initialized = True
        _project = wandb.run.project
        _pipeline_name = wandb.run.name or "unknown_pipeline"
