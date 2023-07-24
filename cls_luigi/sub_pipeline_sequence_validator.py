from typing import List

import luigi
from luigi.task import flatten

from cls_luigi.inhabitation_task import RepoMeta


class SubPipelineSequenceValidator:
    """
    This validator checks if a pipeline is valid according to a given sub pipeline template.
    Not all tasks in the template must be present in the pipeline. However, the order must be preserved.
    if no sub_pipeline is found (no tasks from the template are in pipeline),
     the pipeline is valid by default (no_sub_pipeline_is_possible).
    """

    def __init__(self,
                 sub_pipeline_template: List[RepoMeta],
                 no_sub_pipeline_is_valid: bool = True
                 ) -> None:

        # we revers the template, because the pipeline is traversed from the end to the beginning
        self.sub_pipeline_template = self.get_reversed_str_template(sub_pipeline_template)
        self.no_sub_pipeline_is_valid = no_sub_pipeline_is_valid

    def validate(self,
                 pipeline: luigi.Task
                 ) -> bool:

        sub_pipeline = self.get_sub_pipelines(pipeline)
        return self.is_valid(sub_pipeline)

    def is_valid(self,
                 sub_pipeline: list
                 ) -> bool:

        if len(sub_pipeline) == 0:
            return self.no_sub_pipeline_is_valid

        elif len(sub_pipeline) == 1 and sub_pipeline[0] in self.sub_pipeline_template:
            return True
        else:
            last_task_index = None
            for task in sub_pipeline:
                if last_task_index is None:
                    last_task_index = self.sub_pipeline_template.index(task)
                else:
                    current_task_index = self.sub_pipeline_template.index(task)
                    if current_task_index > last_task_index:
                        last_task_index = current_task_index
                    else:
                        return False
            return True

    def get_sub_pipelines(self,
                          task: luigi.Task
                          ) -> list:

        sub_pipelines = []
        task_name = task.__class__.__name__
        if task_name not in sub_pipelines and task_name in self.sub_pipeline_template:
            sub_pipelines.append(task_name)
        for child in flatten(task.requires()):
            sub_pipelines.extend(self.get_sub_pipelines(child))
        return sub_pipelines

    @staticmethod
    def get_reversed_str_template(template: list) -> list:

        str_template = list(map(lambda x: x.__name__, template))
        reversed_str_template = list(reversed(str_template))
        return reversed_str_template
