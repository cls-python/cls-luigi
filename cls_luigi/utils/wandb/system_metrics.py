import threading
import psutil
import time
from cls_luigi.utils.wandb.core import wandb_log

try:
    import GPUtil
except ImportError:
    GPUtil = None

try:
    import pyamdgpuinfo
except ImportError:
    pyamdgpuinfo = None

# Create a module event to control the logging thread
_stop_event = threading.Event()


def set_stop_even():
    global _stop_event
    _stop_event.set()


def reset_stop_event():
    global _stop_event
    _stop_event = threading.Event()  # Create a new event, which is unset (False)


def start_logging_metrics():
    """Start logging system metrics in a separate thread."""
    process = psutil.Process()
    log_thread = threading.Thread(
        target=_log_system_metrics, args=(process,), daemon=True
    )
    log_thread.start()
    return log_thread  # Return the thread for later use


def _get_nvidia_metrics():
    """Get GPU metrics for NVIDIA GPUs."""
    gpu_usage = {}
    # TODO
    gpu_usage["NVIDIA"] = [0]

    return gpu_usage


def _get_amd_metrics():
    """Get GPU metrics for AMD GPUs."""
    gpu_usage = {}
    # TODO
    gpu_usage["AMD"] = [0]

    return gpu_usage


def _log_system_metrics(process):
    """Log CPU, memory, and GPU usage for the current process and its children."""
    while not _stop_event.is_set():  # Check if the stop event is set
        try:
            # CPU usage
            cpu_usage = process.cpu_percent(
                interval=1
            )  # Total CPU usage including children

            # Memory usage
            memory_info = process.memory_info()
            memory_usage = (
                memory_info.rss / psutil.virtual_memory().total * 100
            )  # RSS as a percentage of total memory

            # Get GPU usage if applicable
            gpu_metrics = {}
            if GPUtil:
                gpu_metrics["NVIDIA"] = _get_nvidia_metrics()
            if pyamdgpuinfo:
                gpu_metrics["AMD"] = _get_amd_metrics()

            # Log the metrics using wandb_log with metrics data type
            wandb_log(
                {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "gpu_metrics_nvidia": gpu_metrics.get("NVIDIA", 0),
                    "gpu_metrics_amd": gpu_metrics.get("AMD", 0),
                }
            )

            # Optional: Print to console for real-time monitoring
            print(
                f"CPU Usage: {cpu_usage}, Memory Usage: {memory_usage}, GPU Metrics: {gpu_metrics}"
            )

            time.sleep(1)  # Adjust the sleep time as needed

        except Exception as e:
            # Log any errors that occur during metric collection
            wandb_log({"system_metrics_error": str(e)})
            # Optionally break the loop or continue
            break
