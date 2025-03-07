import re
import time
from typing import Any, Dict, Optional, List, Union
from luigi import Task, build
from cls.debug_util import deep_str
from cls_luigi.utils.wandb.core import wandb_init, wandb_finish
from cls_luigi.utils.wandb.system_metrics import (
    start_logging_metrics,
    set_stop_even,
    reset_stop_event,
)
from pathlib import Path

from cls_luigi.cls_tasks import LuigiCombinator


class WandbTask(Task, LuigiCombinator):
    """Base class for tasks that use wandb logging."""

    abstract = True

    def run(self):
        """Override this method in derived classes to implement task logic"""
        raise NotImplementedError()

    def on_failure(self, exception):
        wandb_finish()
        return super().on_failure(exception)


def _extract_task_classes(input_str):
    task_classes = []

    main_class_pattern = r"(\w+)\("
    main_class_match = re.search(main_class_pattern, input_str)
    if main_class_match:
        task_classes.append(main_class_match.group(1))  # Get the class name
    pattern = r'"task_class":\s*"([^"]+)"'
    matches = re.findall(pattern, input_str)
    task_classes.extend(matches)
    return "_".join(task_classes)


def run_luigi_pipeline_with_wandb(
    pipeline,
    project_name: str,
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
    dir: Optional[Union[str, Path]] = None,
    id: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    mode: Optional[str] = None,
    save_code: Optional[bool] = None,
    resume: Optional[Union[bool, str]] = None,
    debug_print: bool = False,
) -> None:
    """Run a Luigi pipeline with Weights & Biases tracking.

    Args:
        pipeline: The Luigi task to run
        project_name: Name of the W&B project to log to
        config: Optional dictionary of configuration parameters to log
        entity: Optional username or team name where you're sending runs
        dir: Directory where W&B files will be stored
        id: Unique identifier for the run
        notes: Notes about the run to be stored with the run data
        tags: Tags to assign to the run
        group: Group ID to organize runs together
        job_type: The type of job running
        mode: Can be "online", "offline" or "disabled"
        save_code: Save the main script or notebook and create a code artifact
        resume: Resume a previous run
        debug_print: Whether to print debug information
    """
    pipeline_name = (
        str(_extract_task_classes(str(pipeline))) + "_" + time.strftime("%Y%m%d-%H%M%S")
    )

    # Initialize W&B with the new parameter structure
    wandb_init(
        project_name=project_name,
        pipeline_name=pipeline_name,
        config=config,
        entity=entity,
        dir=dir,
        id=id,
        notes=notes,
        tags=tags,
        group=group,
        job_type=job_type,
        mode=mode,
        save_code=save_code,
        resume=resume,
    )
    log_thread = start_logging_metrics()

    # Only print debug information if debug_print is True
    if debug_print:
        print("==============")
        print(deep_str(pipeline))
        print("\n")

    # Run the Luigi pipeline
    build([pipeline], local_scheduler=True, detailed_summary=True)

    # Only print debug information if debug_print is True
    if debug_print:
        print("\n")
        print("===============")

    # Clean up and finish the W&B run
    set_stop_even()
    log_thread.join()
    wandb_finish()
    reset_stop_event()
