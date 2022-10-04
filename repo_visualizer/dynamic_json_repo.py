from luigi.task import flatten
from repo_visualizer.json_io import load_json, dump_json

CONFIG = "config.json"


class DynamicJSONRepo:
    """
    Constructs a single json pipeline representation from all pipelines, that are produced from the FiniteCombinatoryLogic.
    The pipeline doesn't include any abstract tasks.

    The status of each task in the pipelines will be updated from within JavaScript-app via : http://localhost:8082/api/task_list .
    """

    def __init__(self, cls_results):
        self.cls_results = cls_results
        self.dynamic_pipeline_dict = {}
        self._construct_dynamic_pipeline_dict()

    @staticmethod
    def _prettify_task_name(task):
        listed_task_id = task.task_id.split("_")
        return listed_task_id[0] + "_" + listed_task_id[-1]  # @ [-1] is the hash of the task

    def _construct_dynamic_pipeline_dict(self):
        def populate_dynamic_pipelines_dict(task):
            name = self._prettify_task_name(task)

            if name not in self.dynamic_pipeline_dict:
                self.dynamic_pipeline_dict[name] = {
                    "inputQueue": [],
                    "status": "NOTASSIGNED",
                    "luigiName": task.task_id,
                    # Task-id from luigi itself. It will be shown @ http://localhost:8082/api/task_list
                }
            children = flatten(task.requires())
            for child in children:
                if self._prettify_task_name(child) not in self.dynamic_pipeline_dict[name]["inputQueue"]:
                    self.dynamic_pipeline_dict[name]["inputQueue"] = self.dynamic_pipeline_dict[name]["inputQueue"] + [
                        self._prettify_task_name(child)]
                populate_dynamic_pipelines_dict(child)

        for r in self.cls_results:
            populate_dynamic_pipelines_dict(r)

    def dump_dynamic_pipeline_json(self):
        outfile_name = load_json(CONFIG)['dynamic_repo']
        dump_json(outfile_name, self.dynamic_pipeline_dict)
