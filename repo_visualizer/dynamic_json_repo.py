from luigi.task import flatten
from repo_visualizer.json_io import load_json, dump_json
import time


CONFIG = "config.json"


class DynamicJSONRepo:
    """
    Constructs a single json pipeline representation from all pipelines, that are produced from the FiniteCombinatoryLogic.
    The pipeline doesn't include any abstract tasks.

    The status of each task in the pipelines will be updated from within JavaScript-app via : http://localhost:8082/api/task_list .
    """

    def __init__(self, cls_results):
        self.cls_results = cls_results
        self.dynamic_compressed_pipeline_dict = {}
        self.dynamic_detailed_pipeline_dict = {}
        self._construct_dynamic_pipeline_dict()

    @staticmethod
    def _prettify_task_name(task):
        listed_task_id = task.task_id.split("_")
        return listed_task_id[0] + "_" + listed_task_id[-1]  # @ [-1] is the hash of the task



    def _construct_dynamic_pipeline_dict(self):
        def _get_deps_tree(task, base_dict=None):
            if base_dict is None:
                base_dict = {}
            name = self._prettify_task_name(task)

            task_dict = {"inputQueue": [],
                         "status": "NOTASSIGNED",
                         "luigiName": task.task_id,
                         "createdAt": time.time(),
                         "timeRunning": None,
                         "startTime": None,
                         "lastUpdated": None
                         # Task-id from luigi itself. It will be shown @ http://localhost:8082/api/task_list
                         }

            if name not in base_dict:
                base_dict[name] = task_dict
            children = flatten(task.requires())
            for child in children:
                child_name = self._prettify_task_name(child)
                if child_name not in base_dict[name]["inputQueue"]:
                    base_dict[name]["inputQueue"] = base_dict[name]["inputQueue"] + [child_name]
                _get_deps_tree(child, base_dict)

            return base_dict


        for ix, r in enumerate(self.cls_results):
            pipeline = _get_deps_tree(r)
            self.dynamic_detailed_pipeline_dict[ix] = {}
            self.dynamic_detailed_pipeline_dict[ix].update(pipeline)


    def dump_dynamic_pipeline_json(self):
        config = load_json(CONFIG)
        dump_json(config['dynamic_pipeline'], self.dynamic_detailed_pipeline_dict)
