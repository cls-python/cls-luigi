# core.py
import wandb
import os
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
import warnings
import numbers

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
    """Initialize W&B run with project and pipeline context"""
    global _initialized, _project, _pipeline_name

    if wandb.run is not None:
        raise RuntimeError("W&B already initialized")

    _project = project_name
    _pipeline_name = pipeline_name

    full_config = {"pipeline_name": pipeline_name, **(config or {})}

    wandb.init(project=project_name, name=pipeline_name, config=full_config, **kwargs)
    _initialized = True


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
        - metric: Numerical values or dictionaries with a "value" key containing numbers/strings
        - image: W&B Image objects or dictionaries with a "value" key containing:
            - matplotlib figures
            - numpy arrays (2D or 3D)
            - file paths to image files (.png, .jpg, .jpeg)
        - object3d: W&B Object3D objects or dictionaries with a "value" key containing:
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
        wandb_log({"accuracy": {"value": 0.95, "type": "metric"}})

        # Log an image directly
        wandb_log({"image": wandb.Image("path/to/image.jpg")})

        # Log an image with a dictionary configuration
        wandb_log({"image": {"value": plt.figure(), "type": "image"}})

        # Log a 3D object
        wandb_log({"model": {"value": "path/to/model.obj", "type": "image"}})

        # Log multiple items at once
        wandb_log({
            "loss": 0.5,
            "accuracy": 0.95,
            "confusion_matrix": wandb.Image(confusion_matrix_fig),
            "embeddings": {"value": embedding_array, "type": "histogram"}
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
                # For videos, ensure they're properly wrapped in wandb.Video if needed
                # if not isinstance(data_dict, wandb.Video):
                #     if isinstance(data_dict, str) and os.path.isfile(data_dict):
                #         data_dict = wandb.Video(data_dict)
                # wandb.log({name: data_dict}, step=step, commit=commit)
                _log_video({name: data_dict}, step=step, commit=commit, sync=sync)
                # TODO

            elif dtype =wandb_log_artifact= "audio":
                # For audio, ensure they're properly wrapped in wandb.Audio if needed
                # if not isinstance(data_dict, wandb.Audio):
                #     if isinstance(data_dict, str) and os.path.isfile(data_dict):
                #         data_dict = wandb.Audio(data_dict)
                # wandb.log({name: data_dict}, step=step, commit=commit, sync=sync)
                _log_audio({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "histogram":
                # For histograms, use the existing _log_histogram function
                _log_histogram({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "table":
                # For tables, use the existing _log_table function
                _log_table({name: data_dict}, step=step, commit=commit, sync=sync)

            elif dtype == "html":
                # For HTML content
                # if not isinstance(data_dict, wandb.Html):
                #     data_dict = wandb.Html(data_dict)
                # wandb.log({name: data_dict}, step=step, commit=commit, sync=sync)
                _log_html({name: data_dict}, step=step, commit=commit, sync=sync)

            else:
                # raise error not supported
                raise ValueError(f"Data type {dtype} not supported")

        except Exception as e:
            warnings.warn(f"Error logging {name} with type {dtype}: {str(e)}")


def wandb_log_artifact(artifact_data: Dict[str, Any],)

def wandb_log_model(
    model: Any, name: str, framework: str = "auto", metadata: Optional[Dict] = None
) -> None:
    """Safe model logging with framework detection"""
    _check_initialized()

    if framework == "auto":
        framework = _detect_framework(model)

    _log_model(
        {
            "model": model,
            "name": name,
            "framework": framework,
            "metadata": metadata or {},
        }
    )


def wandb_log_plot(figure: Any, name: str = "plot") -> None:
    """Safe plot logging with matplotlib and plotly check"""
    _check_initialized()

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
    elif pio and isinstance(figure, dict):  # Assuming Plotly figure is in dict format
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

    # Log the processed media using the _log_media function
    _log_media(processed)


# def wandb_log_table(table: Any, name: str = "table") -> None:
#     """Safe table logging with pandas check"""
#     _check_initialized()

#     # Prepare the table for logging
#     if pd and isinstance(table, pd.DataFrame):
#         # If it's a DataFrame, convert it to a wandb Table
#         wandb_table = wandb.Table(dataframe=table)
#         _log_table({name: wandb_table})
#     elif isinstance(table, wandb.Table):
#         # If it's already a wandb Table, log it directly
#         _log_table({name: table})
#     elif isinstance(table, dict):
#         # If it's a dictionary, pass it directly to _log_table
#         _log_table({name: table})
#     elif isinstance(table, list) and all(isinstance(item, dict) for item in table):
#         # If it's a list of dictionaries, create a wandb Table
#         wandb_table = wandb.Table(data=table)
#         _log_table({name: wandb_table})
#     else:
#         raise ValueError(
#             "Input must be a Pandas DataFrame, a wandb.Table, a dictionary, or a list of dictionaries."
#         )


# def wandb_log_histogram(data: Any, name: str = "histogram") -> None:
#     _check_initialized()

#     # Prepare the histogram data
#     if isinstance(data, wandb.Histogram):
#         # If it's already a wandb Histogram, log it directly
#         _log_histogram({name: data})
#     elif isinstance(data, list) and all(
#         isinstance(item, (int, float)) for item in data
#     ):
#         # If it's a list containing only numeric values, pass it to _log_histogram
#         _log_histogram({name: data})
#     elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
#         # If it's a NumPy array containing numeric values, pass it to _log_histogram
#         _log_histogram({name: data})
#     else:
#         raise ValueError(
#             "Input must be a wandb.Histogram, a list of numeric values, or a numpy array of numeric values."
#         )


def wandb_finish() -> None:
    """Finalize W&B run"""
    global _initialized
    if _initialized:
        wandb.finish()
        _initialized = False


def wandb_get_status() -> Dict[str, Any]:
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
    """Log metric data to wandb"""
    for name, metric_data in metric.items():
        if "metadata" in metric_data:
            wandb.log(
                {name: metric_data["value"], **metric_data["metadata"]},
                step=step,
                commit=commit,
                sync=sync,
            )
        else:
            wandb.log({name: metric_data["value"]}, step=step, commit=commit, sync=sync)


def _log_histogram(
    histogram_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    """Log histogram data to wandb"""
    processed = {}
    for key, value in histogram_data.items():
        if isinstance(value, wandb.Histogram):
            processed[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (int, float)) for item in value
        ):
            # Create a wandb Histogram from the list of numeric values
            processed[key] = wandb.Histogram(value)
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            # Create a wandb Histogram from the NumPy array
            processed[key] = wandb.Histogram(value.tolist())  # Convert to list
        else:
            raise ValueError("Unsupported histogram format.")

    # Log the processed histograms
    wandb.log(processed)


def _log_table(
    table: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    processed = {}
    for key, value in table.items():
        if pd and isinstance(value, pd.DataFrame):
            processed[key] = wandb.Table(dataframe=value)
        elif isinstance(value, wandb.Table):
            processed[key] = value
        elif isinstance(value, dict):
            # Convert the dictionary to a wandb Table
            processed[key] = wandb.Table(
                data=[value]
            )  # Assuming a single row dictionary
        else:
            raise ValueError("Unsupported table format.")

    # Log the processed tables
    wandb.log(processed)


def _log_image(
    image_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    """Log image data to wandb

    Args:
        image_data: Dictionary mapping names to image data
        step: Optional step for logging
        commit: Whether to commit the log immediately
        sync: Whether to sync with wandb server immediately

    Examples:
        _log_image({"image": wandb.Image("path/to/image.jpg")})
        _log_image({"image": {"value": plt.figure()}})
        _log_image({"image": {"value": np.zeros((100, 100))}})
        _log_image({"image": {"value": "path/to/image.jpg"}})
        _log_image({"3d_object": {"value": np.zeros((100, 100, 3))}})
        _log_image({"image": {"value": plt.figure(), "caption": "My figure", "metadata": {"key": "value"}}})
    """
    processed = {}

    for key, value in image_data.items():
        # If value is already a wandb.Image or wandb.Object3D, use it directly
        if isinstance(value, (wandb.Image, wandb.Object3D)):
            processed[key] = value
            continue

        # If value is a dictionary with a "value" key, process the inner value
        if isinstance(value, dict) and "value" in value:
            inner_value = value["value"]

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

            # Handle numpy array
            elif np is not None and isinstance(inner_value, np.ndarray):
                if inner_value.ndim in [2, 3]:
                    # Check if it's a 3D object (3D array with 3 channels for RGB)
                    if inner_value.ndim == 3 and inner_value.shape[2] == 3:
                        if img_kwargs:
                            processed[key] = wandb.Object3D(inner_value, **img_kwargs)
                        else:
                            processed[key] = wandb.Object3D(inner_value)

                    else:
                        if img_kwargs:
                            processed[key] = wandb.Image(inner_value, **img_kwargs)
                        else:
                            processed[key] = wandb.Image(inner_value)

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
            else:
                warnings.warn(f"Unsupported image format for key {key}. Skipping.")
        else:
            warnings.warn(f"Unsupported image format for key {key}. Skipping.")

    if processed:
        wandb.log(processed, step=step, commit=commit, sync=sync)


def _log_video(
    video_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    processed = {}

def _log_audio(audio_data: Dict[str, Any], step: Optional[int] = None, commit: bool = True, sync: bool = True) -> None:
    processed = {}

def _log_html(html_data: Dict[str, Any], step: Optional[int] = None, commit: bool = True, sync: bool = True) -> None:
    processed = {}

def _log_media(
    media_data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    sync: bool = True,
) -> None:
    processed = {}
    for key, value in media_data.items():
        if plt and isinstance(value, plt.Figure):
            processed[key] = wandb.Image(value)
            plt.close(value)
        elif np is not None and isinstance(value, np.ndarray):
            if value.ndim in [2, 3]:
                processed[key] = wandb.Image(value)
            elif value.ndim == 3 and value.shape[2] == 3:
                processed[key] = wandb.Object3D(value)
        elif isinstance(value, (str, Path)):
            path = Path(value)
            suffix = path.suffix.lower()
            if suffix in [".png", ".jpg", ".jpeg"]:
                processed[key] = wandb.Image(str(value))
            elif suffix in [".mp4", ".avi"]:
                processed[key] = wandb.Video(str(value))
            elif suffix in [".wav", ".mp3"]:
                processed[key] = wandb.Audio(str(value))
        else:
            processed[key] = value
    wandb.log(processed)


def _log_artifact(
    artifact_data: Dict[str, Any]
) -> None:
    for name, details in artifact_data.items():
        path = details["path"]
        art_type = details.get("type", "file")
        metadata = details.get("metadata", {})

        artifact = wandb.Artifact(name=name, type=art_type)
        path = Path(path)

        if path.is_dir():
            artifact.add_dir(str(path))
            metadata["file_count"] = len(list(path.rglob("*")))
        else:
            artifact.add_file(str(path))
            metadata["file_size"] = os.path.getsize(path)

        artifact.metadata.update(
            {
                "pipeline": _pipeline_name,
                "logged_at": datetime.now().isoformat(),
                **metadata,
            }
        )
        wandb.log_artifact(artifact)


def _log_model(
    model_data: Dict[str, Any]
) -> None:
    model = model_data["model"]
    name = model_data["name"]
    framework = model_data.get("framework", "auto")
    metadata = model_data.get("metadata", {})

    model_dir = Path(wandb.run.dir) / "models"
    model_dir.mkdir(exist_ok=True)

    # Auto-detect framework if not specified
    if framework == "auto":
        if torch is not None and hasattr(model, "state_dict"):
            framework = "pytorch"
        elif keras is not None and hasattr(model, "save"):
            framework = "tensorflow"
        elif xgb is not None and hasattr(model, "save_model"):
            framework = "xgboost"
        elif joblib is not None:
            framework = "generic"
        else:
            raise ValueError("Could not auto-detect model framework")

    # Save model based on framework
    if framework == "pytorch":
        if torch is None:
            raise ImportError("PyTorch required for model logging")
        path = model_dir / f"{name}.pt"
        torch.save(model.state_dict(), path)
    elif framework == "tensorflow":
        if keras is None:
            raise ImportError("TensorFlow required for model logging")
        path = model_dir / name
        model.save(path)
    elif framework == "xgboost":
        if xgb is None:
            raise ImportError("XGBoost required for model logging")
        path = model_dir / f"{name}.xgb"
        model.save_model(str(path))
    elif framework == "lightgbm":
        try:
            import lightgbm as lgb

            path = model_dir / f"{name}.lgb"
            model.save_model(str(path))
        except ImportError:
            raise ImportError("LightGBM required for model logging")
    elif framework == "scikit-learn":
        try:
            from sklearn.base import BaseEstimator

            if not isinstance(model, BaseEstimator):
                raise ValueError("Not a scikit-learn model")
            path = model_dir / f"{name}.pkl"
            joblib.dump(model, path)
        except ImportError:
            raise ImportError("Scikit-learn required for model logging")
    else:
        if joblib is None:
            raise ImportError("Joblib required for generic model logging")
        path = model_dir / f"{name}.pkl"
        joblib.dump(model, path)

    artifact = wandb.Artifact(name, type="model")
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))

    artifact.metadata.update(
        {"framework": framework, "pipeline": _pipeline_name, **metadata}
    )
    wandb.log_artifact(artifact)


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
    """
    Infer the data type for wandb logging based on the content of the data dictionary.

    This function checks for the following data types:
    - metric: Numerical values that represent metrics
    - image: Image data (wandb.Image objects)
    - video: Video data (wandb.Video objects)
    - audio: Audio data (wandb.Audio objects)
    - histogram: Histogram data (wandb.Histogram, lists of numbers, or numpy arrays)
    - table: Table data (wandb.Table, pandas DataFrame, or list of dictionaries)
    - html: HTML content for rendering
    """
    # Check for wandb specific objects
    if any(isinstance(v, wandb.Image) for v in data.values()):
        return "image"

    # Check for image data in the "value" key
    if "value" in data:
        value = data["value"]
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

    if any(isinstance(v, wandb.Audio) for v in data.values()):
        return "audio"

    if any(isinstance(v, wandb.Html) for v in data.values()):
        return "html"

    if any(isinstance(v, wandb.Table) for v in data.values()):
        return "table"

    if any(isinstance(v, wandb.Histogram) for v in data.values()):
        return "histogram"

    # Check for pandas DataFrame (if pandas is available)
    if pd and any(isinstance(v, pd.DataFrame) for v in data.values()):
        return "table"

    # Check for histogram data (list of numbers or numpy array)
    if any(
        (isinstance(v, list) and all(isinstance(item, (int, float)) for item in v))
        or (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number))
        for v in data.values()
    ):
        return "histogram"

    # Check for table data (list of dictionaries)
    if any(
        isinstance(v, list) and all(isinstance(item, dict) for item in v)
        for v in data.values()
    ):
        return "table"

    # Default to metric for any other type
    return (
        "metric"
        if "value" in data.values()
        and (
            isinstance(data["value"], (numbers.Number, str))
            or (np and isinstance(data["value"], np.number))
        )
        else "none"
    )


def _check_initialized() -> None:
    if not _initialized:
        raise RuntimeError("W&B not initialized. Call init_wandb() first")
