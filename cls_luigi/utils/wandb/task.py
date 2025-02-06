from ..cls_tasks import LuigiCombinator
from luigi import Task
from .core import wandb_finish

class WandbTask(Task, LuigiCombinator):
    """Base class for tasks that use wandb logging."""

    abstract = True

    def run(self):
        """Override this method in derived classes to implement task logic"""
        raise NotImplementedError()

    def on_failure(self, exception):
        wandb_finish()
        return super().on_failure(exception)