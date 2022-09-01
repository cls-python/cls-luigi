import json
from os.path import join
from luigi.task import flatten


class DynamicJSONRepo:

    """
    Constructs a single json pipeline representation from all pipelines, that are produced from the FiniteCombinatoryLogic.
    The pipeline doesn't include any abstract tasks.

    The status of each task in the pipelines will be from within JavaScript via : http://localhost:8082/api/task_list updated.
    """

    def __init__(self, cls_results):
        self.path = None
        self.cls_results = cls_results
        self.dynamic_pipeline_dict = {}

    def prettify_task_name(self, task):
        listed_task_id = task.task_id.split("_")
        return listed_task_id[0] + "_" + listed_task_id[-1]  # @ [-1] is the hash of the task

    def construct_dynamic_pipeline_dict(self):
        def get_deps_tree(task):
            name = self.prettify_task_name(task)

            self.dynamic_pipeline_dict[name] = {
                "inputQueue": [],
                "status": "PENDING",
                "luigiName": task.task_id,  # Task-id from luigi itself. It will be shown @ http://localhost:8082/api/task_list
                "abstract": False
            }
            children = flatten(task.requires())
            for child in children:
                if self.prettify_task_name(child) not in self.dynamic_pipeline_dict[name]["inputQueue"]:
                    self.dynamic_pipeline_dict[name]["inputQueue"] = self.dynamic_pipeline_dict[name]["inputQueue"] + [self.prettify_task_name(child)]
                get_deps_tree(child)

        for r in self.cls_results:
            get_deps_tree(r)

    def dump_dynamic_pipeline_dict(self, path=""):
        full_path = join(path, "dynamic_repo.json")
        print(full_path)
        with open(full_path, "w+") as r:
            json.dump(self.dynamic_pipeline_dict, r, indent=4)










