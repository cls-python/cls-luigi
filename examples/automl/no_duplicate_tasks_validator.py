import luigi
from luigi.task import flatten
from typing import List


class NoDuplicateTasksValidator(object):

    def __init__(self):
        pass

    def validate(self, task: luigi.Task) -> bool:
        components = self._get_single_components(task)
        components = list(map(lambda component: component.task_family, components))
        return len(components) == len(set(components))

    def _get_single_components(self, task: luigi.Task) -> List[luigi.Task]:
        pipeline = [task]
        children = flatten(task.requires())

        if children:
            for child in children:
                if child not in pipeline:
                    pipeline.extend(self._get_single_components(child))
        return pipeline
