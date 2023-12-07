from typing import List

import luigi
from luigi.task import flatten

from cls_luigi.inhabitation_task import RepoMeta


class AutoMLPipelineValidator(object):
    def __init__(self, forbidden_tasks: List[List[RepoMeta]]) -> None:
        self.forbidden_tasks = forbidden_tasks
        self.map_forbidden_tasks_to_family_name()

    def validate(self, pipeline: luigi.Task) -> bool:
        pipeline = self.get_pipeline_as_list(pipeline)
        return self._check_subset(pipeline, self.forbidden_tasks)

    def map_forbidden_tasks_to_family_name(self) -> None:
        for ix, collection in enumerate(self.forbidden_tasks):
            for ix_2, task in enumerate(collection):
                if isinstance(task, RepoMeta):
                    self.forbidden_tasks[ix][ix_2] = task.task_family
                elif isinstance(task, str):
                    self.forbidden_tasks[ix][ix_2] = task

    @staticmethod
    def _check_subset(pipeline: List[str], subsets: List[str]) -> bool:
        set_pipeline = set(pipeline)
        for s in subsets:
            subset = set(s)
            if subset.issubset(set_pipeline) is True:
                return False
        return True

    def get_pipeline_as_list(self, task: luigi.Task, task_list: List = None) -> List[str]:
        if task_list is None:
            task_list = []

        task_name = task.task_family

        if task_name not in task_list:
            task_list.append(task_name)
        dependencies = flatten(task.requires())
        for dep in dependencies:
            task_list = self.get_pipeline_as_list(dep, task_list)

        return task_list
