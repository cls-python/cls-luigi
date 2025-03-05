# core.py
import wandb
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
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
    base_estimator = None  #

try:
    import plotly.io as pio
    from plotly.graph_objs import Figure
except ImportError:
    pio = None
    Figure = None

# Module-level state
_initialized = False
_project: Optional[str] = None
_pipeline_name: Optional[str] = None


def wandb_init(
    project_name: str,
    pipeline_name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Initialize a Weights & Biases run with project and pipeline context.
    
    This function initializes a new W&B run and sets global variables to track
    the initialization state. It automatically adds the pipeline name to the
    configuration and handles the W&B initialization process.
    
    Args:
        project_name: Name of the W&B project to log to
        pipeline_name: Name of the current pipeline/run
        config: Optional dictionary of configuration parameters to log
        **kwargs: Additional keyword arguments passed directly to wandb.init()
        
    Raises:
        RuntimeError: If W&B is already initialized
        ImportError: If wandb package is not installed
        
    Examples:
        Basic initialization:
        >>> wandb_init("my-project", "training-pipeline")
        
        With configuration:
        >>> wandb_init(
        ...     "my-project", 
        ...     "training-pipeline", 
        ...     config={"learning_rate": 0.001, "batch_size": 32}
        ... )
        
        With additional W&B parameters:
        >>> wandb_init(
        ...     "my-project",
        ...     "training-pipeline",
        ...     tags=["experiment-1", "resnet"],
        ...     notes="Testing improved model architecture"
        ... )
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
        wandb.init(project=project_name, name=pipeline_name, config=full_config, **kwargs)
        _initialized = True
    except Exception as e:
        # Reset global state on failure
        _project = None
        _pipeline_name = None
        raise RuntimeError(f"Failed to initialize W&B: {str(e)}") from e


def wandb_log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
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
        sync: Whether to sync the data with the W&B server immediately.

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
        # Log a simple metric
        wandb_log({"loss": 0.5})

        # Log a metric with a dictionary configuration
        wandb_log({"accuracy": {"data": 0.95, "type": "metric"}})

        # Log an image directly
        wandb_log({"image": wandb.Image("path/to/image.jpg")})

        # Log an image with a dictionary configuration
        wandb_log({"image": {"data": plt.figure(), "type": "image"}})

        # Log a 3D object
        wandb_log({"model": {"data": "path/to/model.obj", "type": "image"}})

        # Log multiple items at once
        wandb_log({
            "loss": 0.5,
            "accuracy": 0.95,
            "confusion_matrix": wandb.Image(confusion_matrix_fig),
            "embeddings": {"data": embedding_array, "type": "histogram"}
        })

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
        wandb.log(direct_log, step=step, commit=commit, sync=sync)

    # Process the remaining items that need special handling
    for name, data_dict in process_log.items():
        # Determine the data type, either from the explicit type or by inference
        dtype = (
            data_dict["type"] if "type" in data_dict else _infer_data_type(data_dict)
        )

        try:
            if dtype == "metric":
                _log_metric({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "image":
                _log_image({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "video":
                _log_video({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "audio":
                _log_audio({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "histogram":
                # For histograms, use the existing _log_histogram function
                _log_histogram({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "table":
                # For tables, use the existing _log_table function
                _log_table({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "html":
                # For HTML content, use the existing _log_html function
                _log_html({name: data_dict}, step=step, commit=commit, sync=sync)

            else:
                # raise error not supported
                raise ValueError(f"Data type {dtype} not supported")

        except Exception as e:
            warnings.warn(f"Error logging {name} with type {dtype}: {str(e)}")


def wandb_log_plot(
    figure: Any,
    name: str = "plot",
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    """Safely log a matplotlib or plotly figure to Weights & Biases.
    
    This function handles different types of plot figures (matplotlib, plotly) and
    converts them to a format that can be logged to W&B. It includes error handling
    to prevent crashes if the figure cannot be processed.
    
    Args:
        figure: The figure to log, can be matplotlib.figure.Figure, plotly.graph_objs.Figure,
               or a plotly figure in dict format
        name: Name to use for the logged figure in W&B
        step: Optional step number for the logged figure
        commit: Whether to commit the figure immediately
        sync: Whether to sync with wandb server immediately
        
    Examples:
        With matplotlib:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3, 4])
        >>> wandb_log_plot(fig, name="training_loss")
        
        With plotly:
        >>> import plotly.express as px
        >>> fig = px.line(x=[0, 1, 2, 3], y=[0, 1, 4, 9])
        >>> wandb_log_plot(fig, name="accuracy_curve", step=10)
    """
    _check_initialized()

    try:
        processed = {}

        if isinstance(figure, wandb.plot.CustomChart):
            wandb.log({name: figure})
            return

        # Check if the figure is a Matplotlib figure
        if plt and isinstance(figure, plt.Figure):
            processed[name] = figure  # Store the Matplotlib figure directly
            plt.close(figure)  # Close the figure after logging

        # Check if the figure is a Plotly Figure object
        elif pio and Figure and isinstance(figure, Figure):
            img_data = pio.to_image(figure, format="png")
            processed[name] = img_data

        # Check if the figure is a Plotly figure in dict format
        elif pio and isinstance(
            figure, dict
        ):  # Assuming Plotly figure is in dict format
            img_data = pio.to_image(figure, format="png")
            processed[name] = img_data

        # Check if the figure is a JSON string
        elif pio and isinstance(figure, str):
            try:
                # Attempt to parse the JSON string to a Plotly figure
                figure_dict = pio.from_json(figure)
                img_data = pio.to_image(figure_dict, format="png")
                processed[name] = img_data
            except Exception as e:
                raise ValueError("Invalid JSON string for Plotly figure.") from e

        else:
            raise ValueError(
                "The figure must be a Matplotlib figure, a Plotly Figure object, or a Plotly figure in dict or JSON format."
            )

        # Log the processed media using the _log_image function
        _log_image(processed, step=step, commit=commit, sync=sync)

    except Exception as e:
        warnings.warn(f"Error logging {name} with type {type(figure)}: {str(e)}")


def wandb_finish() -> None:
    """Finalize the current Weights & Biases run.
    
    This function safely finalizes the current W&B run if one exists and resets
    the global initialization state. It should be called at the end of your
    pipeline or experiment to properly close the W&B session.
    
    Note that this function does nothing if W&B has not been initialized.
    
    Examples:
        >>> # After initializing and logging data
        >>> wandb_init("my-project", "training-pipeline")
        >>> # ... perform logging operations ...
        >>> wandb_finish()  # Properly close the W&B run
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
            "scikit-learn": base_estimator is not None,
            "plotly": pio is not None,
        },
    }


# Internal helper functions


def _log_metric(
    metric: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
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
        sync: Whether to sync with wandb server immediately
        
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
                sync=sync,
            )
        elif "data" in metric_data:
            wandb.log({name: metric_data["data"]}, step=step, commit=commit, sync=sync)
        else:
            warnings.warn(
                f"Unsupported metric data type for key {name}: {type(metric_data)}. Skipping."
            )


def _log_histogram(
    histogram_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
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
        sync: Whether to sync with wandb server immediately
        
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
            wandb.log(processed, step=step, commit=commit, sync=sync)
    except Exception as e:
        warnings.warn(f"Error logging histogram data: {str(e)}")


def _log_table(
    table: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
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
        sync: Whether to sync with wandb server immediately
        
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
            wandb.log(processed, step=step, commit=commit, sync=sync)

    except Exception as e:
        warnings.warn(f"Error logging table data: {str(e)}")


def _log_image(
    image_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
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
        sync: Whether to sync with wandb server immediately
        
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
            wandb.log(processed, step=step, commit=commit, sync=sync)
    except Exception as e:
        warnings.warn(f"Error logging image data: {str(e)}")


def _log_video(
    video_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    """
    Log video data to wandb

    Args:
        video_data: Dictionary mapping names to video data
        step: Optional step for logging
        commit: Whether to commit the log immediately
        sync: Whether to sync with wandb server immediately

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
            wandb.log(processed, step=step, commit=commit, sync=sync)
    except Exception as e:
        warnings.warn(f"Error logging video data: {str(e)}")


def _log_audio(
    audio_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
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
            wandb.log(processed, step=step, commit=commit, sync=sync)

    except Exception as e:
        warnings.warn(f"Error logging audio data: {str(e)}")


def _log_html(
    html_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
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
            wandb.log(processed, step=step, commit=commit, sync=sync)
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
    if base_estimator and isinstance(model, BaseEstimator):
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


def wandb_log_artifact(artifact_data: Dict[str, Any]) -> None:
    """Log an artifact to Weights & Biases.
    
    This function logs artifacts (models, datasets, etc.) to W&B for versioning
    and tracking. It handles different types of artifacts and provides a consistent
    interface for logging them.
    
    Args:
        artifact_data: Dictionary containing artifact information with the following keys:
            - 'name': Name of the artifact (required)
            - 'type': Type of artifact (e.g., 'model', 'dataset') (required)
            - 'description': Optional description of the artifact
            - 'metadata': Optional dictionary of metadata to associate with the artifact
            - 'path': Path to the artifact file or directory (required)
            - 'aliases': Optional list of aliases to apply to the artifact
    
    Examples:
        >>> # Log a model artifact
        >>> wandb_log_artifact({
        ...     'name': 'resnet50_model',
        ...     'type': 'model',
        ...     'description': 'ResNet-50 model trained on ImageNet',
        ...     'path': 'models/resnet50.pth',
        ...     'metadata': {'accuracy': 0.76, 'parameters': 25.6e6},
        ...     'aliases': ['best', 'v1']
        ... })
        
        >>> # Log a dataset artifact
        >>> wandb_log_artifact({
        ...     'name': 'processed_dataset',
        ...     'type': 'dataset',
        ...     'path': 'data/processed/',
        ...     'metadata': {'num_samples': 10000, 'classes': 10}
        ... })
    """
    _check_initialized()
    
    # Validate required fields
    required_fields = ['name', 'type', 'path']
    for field in required_fields:
        if field not in artifact_data:
            raise ValueError(f"Required field '{field}' missing from artifact_data")
    
    # Create the artifact
    artifact = wandb.Artifact(
        name=artifact_data['name'],
        type=artifact_data['type'],
        description=artifact_data.get('description', ''),
        metadata=artifact_data.get('metadata', {})
    )
    
    # Add the file or directory to the artifact
    path = Path(artifact_data['path'])
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    
    # Log the artifact
    aliases = artifact_data.get('aliases', [])
    wandb.log_artifact(artifact, aliases=aliases)


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
