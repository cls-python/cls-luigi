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

from collections.abc import Iterable


class UniqueTaskPipelineValidator(object):

    def __init__(self, unique_abstract_classes):
        self.unique_abstract_classes = unique_abstract_classes

    def validate(self, pipeline):
        traversal = self.dfs(pipeline)
        concrete_cls_map = dict()
        for obj in traversal:
            for cls in self.unique_abstract_classes:
                if isinstance(obj, cls):
                    if cls in concrete_cls_map:
                        concrete_cls = concrete_cls_map[cls]
                        if not isinstance(obj, concrete_cls):
                            return False
                    else:
                        concrete_cls_map[cls] = obj.__class__
        return True

    def dfs(self, start):
        traversal = [start]
        dependencies = start.requires()
        if isinstance(dependencies, dict):
            for dependency in start.requires().values():
                traversal.extend(self.dfs(dependency))
        elif isinstance(dependencies, Iterable):
            for dependency in start.requires():
                traversal.extend(self.dfs(dependency))
        else:
            traversal.extend(self.dfs(dependencies))

        return traversal


