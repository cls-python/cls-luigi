# -*- coding: utf-8 -*-
#
# Apache Software License 2.0
#
# Copyright (c) 2022-2023, Jan Bessai, Anne Meyer, Hadi Kutabi, Daniel Scholtyssek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from luigi.task import flatten
from cls_luigi.visualizer.json_io import dump_json
import time
import os

VIS = os.path.dirname(os.path.abspath(__file__))


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

        self.dynamic_pipeline_json = os.path.join(
            VIS, "static", "dynamic_pipeline.json"
        )

        if os.path.exists(self.dynamic_pipeline_json):
            os.remove(self.dynamic_pipeline_json)

    @staticmethod
    def _prettify_task_name(task):
        listed_task_id = task.task_id.split("_")
        return (
            listed_task_id[0] + "_" + listed_task_id[-1]
        )  # @ [-1] is the hash of the task

    def _construct_dynamic_pipeline_dict(self):
        def _get_deps_tree(task, base_dict=None):
            if base_dict is None:
                base_dict = {}
            name = self._prettify_task_name(task)

            task_dict = {
                "inputQueue": [],
                "status": "NOTASSIGNED",
                "luigiName": task.task_id,
                "createdAt": time.time(),
                "timeRunning": None,
                "startTime": None,
                "lastUpdated": None,
                "processingTime": None,
                # Task-id from luigi itself. It will be shown @ http://localhost:8082/api/task_list
            }

            if name not in base_dict:
                base_dict[name] = task_dict
            children = flatten(task.requires())
            for child in children:
                child_name = self._prettify_task_name(child)
                if child_name not in base_dict[name]["inputQueue"]:
                    base_dict[name]["inputQueue"] = base_dict[name]["inputQueue"] + [
                        child_name
                    ]
                _get_deps_tree(child, base_dict)

            return base_dict

        for ix, r in enumerate(self.cls_results):
            pipeline = _get_deps_tree(r)
            self.dynamic_detailed_pipeline_dict[ix] = {}
            self.dynamic_detailed_pipeline_dict[ix].update(pipeline)

    def dump_dynamic_pipeline_json(self):
        outfile_name = self.dynamic_pipeline_json
        dump_json(outfile_name, self.dynamic_detailed_pipeline_dict)
