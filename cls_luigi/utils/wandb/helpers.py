import re
import time
from typing import Any, Dict, Optional
from luigi import Task, build
from cls.debug_util import deep_str
from cls_luigi.utils.wandb.core import wandb_init, wandb_finish
from cls_luigi.utils.wandb.system_metrics import start_logging_metrics, set_stop_even, reset_stop_event

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


def run_luigi_pipeline_with_wandb(pipeline, project_name: str, config: Optional[Dict[str, Any]] = None, debug_print=False, **kwargs) -> None:
    pipeline_name = str(_extract_task_classes(str (pipeline))) + "_" + time.strftime("%Y%m%d-%H%M%S")
    wandb_init(project_name, pipeline_name, config, **kwargs)
    log_thread = start_logging_metrics()
    print("==============")
    print(deep_str(pipeline))
    print("\n")
    build([pipeline], local_scheduler=True, detailed_summary=True)
    print("\n")
    print("===============")
    set_stop_even()
    log_thread.join()
    wandb_finish()
    reset_stop_event()
