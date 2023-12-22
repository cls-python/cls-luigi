from typing import List, Set

import luigi
from luigi.task import flatten

from examples.automl.implementations.forbidden import FORBIDDEN


class NotForbiddenValidator(object):
    def __init__(self, forbidden_tasks: List[Set[str]] = FORBIDDEN) -> None:
        self.forbidden_tasks = forbidden_tasks

    def validate(self, pipeline: luigi.Task) -> bool:
        pipeline = self.get_pipeline_as_list(pipeline)
        return self._check_subset(pipeline, self.forbidden_tasks)

    @staticmethod
    def _check_subset(pipeline: List[str], subsets: Set[str]) -> bool:
        pipeline = set(pipeline)
        for s in subsets:
            if s.issubset(pipeline) is True:
                return False
        return True

    def get_pipeline_as_list(self, task: luigi.Task) -> List[str]:
        pipeline = [task.task_family]
        children = flatten(task.requires())

        if children:
            for child in children:
                if child.task_family not in pipeline:
                    pipeline.extend(self.get_pipeline_as_list(child))
        return pipeline
