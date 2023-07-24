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


class SingleInstancePerTaskInPipelineValidator:
    """
    This validator checks if there is only one instance of a task in the pipeline.
    """
    def __init__(self, classes_to_check):
        self.classes_to_check = classes_to_check

    def validate(self, pipeline):
        unique_tasks = self.get_unique_tasks(pipeline)

        for cls in self.classes_to_check:
            checked_tasks = list(filter(lambda x: isinstance(x, cls), unique_tasks))
            if len(checked_tasks) > 1:
                return False
        return True

    def get_unique_tasks(self, end_task):
        unique_tasks = [end_task]
        dependencies = flatten(end_task.requires())

        for dep in dependencies:
            unique_tasks.extend(self.get_unique_tasks(dep))

        return list(set(unique_tasks))
