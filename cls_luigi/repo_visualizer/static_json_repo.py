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

import cls.types
from cls_luigi.inhabitation_task import RepoMeta
from cls_luigi.repo_visualizer.json_io import load_json, dump_json

from cls.types import Constructor
import os

VIS = os.path.dirname(os.path.abspath(__file__))

CONFIG = "config.json"


class StaticJSONRepo:
    """
    Creates a json representation of all components in repository, their types,
    their concrete implementations or config indexes, and their dependencies.

    usage:
      StaticJSONRepo(RepoMeta).dump_static_repo_json()
    """

    def __init__(self, repo_meta):
        self.repo_meta = repo_meta
        self.repository = repo_meta.repository
        self.concrete_to_abst_mapper = {}

        self.output_dict = {}
        self.add_tasks_and_concreate_implementations()
        self.add_dependencies_and_indexed_tasks()

        self.static_pipeline_json = os.path.join(VIS, 'static_pipeline.json')
        if os.path.exists(self.static_pipeline_json):
            os.remove(self.static_pipeline_json)

    @staticmethod
    def _prettify_name(name):
        return name.split(".")[-1]

    def add_tasks_and_concreate_implementations(self):
        for k, v in self.repo_meta.subtypes.items():
            if (k.tpe.abstract is not None) and (k.tpe.abstract is True):
                pretty_task_name = self._prettify_name(k.cls_tpe)
                self.output_dict[pretty_task_name] = {"abstract": True}

        for k, v in self.repo_meta.subtypes.items():
            if (k.tpe.abstract is not None) and (k.tpe.abstract is False):
                pretty_task_name = self._prettify_name(k.cls_tpe)

                if v:
                    if list(v)[0].tpe.abstract is None or (not v):
                        self.output_dict[pretty_task_name] = {"abstract": False}

                    elif list(v)[0].tpe.abstract is True:
                        pretty_abst_task = self._prettify_name(list(v)[0].cls_tpe)
                        if "concreteImplementations" not in self.output_dict[pretty_abst_task]:
                            self.output_dict[pretty_abst_task]["concreteImplementations"] = []

                        self.output_dict[pretty_abst_task]["concreteImplementations"] += [pretty_task_name]
                        self.concrete_to_abst_mapper[pretty_task_name] = pretty_abst_task
                else:
                    self.output_dict[pretty_task_name] = {"abstract": False}

    def add_dependencies_and_indexed_tasks(self):
        for k, v in self.repository.items():
            if not isinstance(k, RepoMeta.ClassIndex):
                if k.cls.abstract is not None:
                    if isinstance(v, cls.types.Intersection):
                        pretty_task = self._prettify_name(k.name)
                        intersection = v
                        inter_org = list(intersection.organized)
                        for ix, i in enumerate(inter_org):
                            path = i.path
                            index = path[0][0].name.at_index
                            indexed_t_name = path[0][1].name.cls_tpe
                            pretty_indexed_t_name = self._prettify_name(indexed_t_name)
                            if "configIndexes" not in self.output_dict[pretty_task]:
                                self.output_dict[pretty_task]["configIndexes"] = {}
                            self.output_dict[pretty_task]["configIndexes"][index] = [pretty_indexed_t_name]

                            if pretty_indexed_t_name in self.concrete_to_abst_mapper:
                                pretty_indexed_t_name = self.concrete_to_abst_mapper[pretty_indexed_t_name]

                            if "inputQueue" not in self.output_dict[pretty_task]:
                                self.output_dict[pretty_task]["inputQueue"] = []

                            self.output_dict[pretty_task]["inputQueue"] += [pretty_indexed_t_name]

                    if not isinstance(v, Constructor):
                        pretty_task = self._prettify_name(k.name)

                        predecessors = v.path
                        if predecessors:
                            for p in predecessors[0]:
                                pretty_p = self._prettify_name(p.name.cls_tpe)
                                if pretty_p in self.concrete_to_abst_mapper:
                                    pretty_p = self.concrete_to_abst_mapper[pretty_p]
                                if pretty_task in self.concrete_to_abst_mapper:
                                    pretty_task = self.concrete_to_abst_mapper[pretty_task]
                                if "inputQueue" not in self.output_dict[pretty_task]:
                                    self.output_dict[pretty_task]["inputQueue"] = []
                                self.output_dict[pretty_task]["inputQueue"] += [pretty_p]

    def dump_static_repo_json(self):
        outfile_name = self.static_pipeline_json
        dump_json(outfile_name, self.output_dict)
