# core.py
import wandb
import os
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
import warnings

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
    project: str, pipeline_name: str, config: Optional[Dict[str, Any]] = None, **kwargs
) -> None:
    """Initialize W&B run with project and pipeline context"""
    global _initialized, _project, _pipeline_name

    if wandb.run is not None:
        raise RuntimeError("W&B already initialized")

    _project = project
    _pipeline_name = pipeline_name

    full_config = {"pipeline_name": pipeline_name, **(config or {})}

    wandb.init(project=project, config=full_config, **kwargs)
    _initialized = True


def wandb_log(
    data: Dict[str, Any],
    step: Optional[int] = None,
    commit: bool = True,
    data_type: Optional[str] = None,
) -> None:
    """Universal logging function with dependency-safe handling"""
    _check_initialized()
    dtype = data_type or _infer_data_type(data)

    if dtype == "metrics":
        wandb.log(data, step=step, commit=commit)
    elif dtype == "params":
        wandb.config.update(data)
    elif dtype == "media":
        _log_media(data)
    elif dtype == "artifact":
        _log_artifact(data)
    elif dtype == "model":
        _log_model(data)
    elif dtype == "table":
        _log_table(data)
    elif dtype == "histogram":
        _log_histogram(data)
    else:
        warnings.warn(f"Unsupported data type: {dtype}")


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


def wandb_log_table(table: Any, name: str = "table") -> None:
    """Safe table logging with pandas check"""
    _check_initialized()

    # Prepare the table for logging
    if pd and isinstance(table, pd.DataFrame):
        # If it's a DataFrame, convert it to a wandb Table
        wandb_table = wandb.Table(dataframe=table)
        _log_table({name: wandb_table})
    elif isinstance(table, dict):
        # If it's a dictionary, pass it directly to _log_table
        _log_table({name: table})
    elif isinstance(table, list) and all(isinstance(item, dict) for item in table):
        # If it's a list of dictionaries, create a wandb Table
        wandb_table = wandb.Table(data=table)
        _log_table({name: wandb_table})
    else:
        raise ValueError(
            "Input must be a Pandas DataFrame, a dictionary, or a list of dictionaries."
        )


def wandb_log_histogram(data: Any, name: str = "histogram") -> None:
    _check_initialized()

    # Prepare the histogram data
    if isinstance(data, wandb.Histogram):
        # If it's already a wandb Histogram, log it directly
        _log_histogram({name: data})
    elif isinstance(data, list) and all(
        isinstance(item, (int, float)) for item in data
    ):
        # If it's a list containing only numeric values, pass it to _log_histogram
        _log_histogram({name: data})
    elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        # If it's a NumPy array containing numeric values, pass it to _log_histogram
        _log_histogram({name: data})
    else:
        raise ValueError(
            "Input must be a wandb.Histogram, a list of numeric values, or a numpy array of numeric values."
        )


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
            "matplotlib": plt is not None,
            "pytorch": torch is not None,
            "tensorflow": keras is not None,
        },
    }


# Internal helper functions


def _log_histogram(histogram_data: Dict[str, Any]) -> None:
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


def _log_table(table: Dict[str, Any]) -> None:
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


def _log_media(media_data: Dict[str, Any]) -> None:
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


def _log_artifact(artifact_data: Dict[str, Any]) -> None:
    for name, details in artifact_data.items():
        path = details["path"]
        art_type = details.get("type", "dataset")
        metadata = details.get("metadata", {})

        artifact = wandb.Artifact(name, type=art_type)
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


def _log_model(model_data: Dict[str, Any]) -> None:
    model = model_data["model"]
    name = model_data["name"]
    framework = model_data["framework"]
    metadata = model_data.get("metadata", {})

    model_dir = Path(wandb.run.dir) / "models"
    model_dir.mkdir(exist_ok=True)

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
    if any(
        isinstance(v, (wandb.Image, wandb.Video, wandb.Audio)) for v in data.values()
    ):
        return "media"
    if any("model" in k.lower() for k in data.keys()):
        return "model"
    if any("path" in k.lower() for k in data.keys()):
        return "artifact"
    if any(
        isinstance(v, wandb.Table)
        or (pd and isinstance(v, pd.DataFrame))
        or (isinstance(v, list) and all(isinstance(item, dict) for item in v))
        for v in data.values()
    ):
        return "table"
    if any(
        isinstance(v, wandb.Histogram)
        or (isinstance(v, list) and all(isinstance(item, (int, float)) for item in v))
        or (isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number))
        for v in data.values()
    ):
        return "histogram"
    if all(
        isinstance(v, (int, float, bool, complex, str, dict, list))
        for v in data.values()
    ):
        return "metrics"
    return "params"


def _check_initialized() -> None:
    if not _initialized:
        raise RuntimeError("W&B not initialized. Call init_wandb() first")
