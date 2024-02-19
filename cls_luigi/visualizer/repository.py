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

from luigi import Task, WrapperTask

from cls_luigi.visualizer.json_io import dump_json
from cls.types import Constructor, Intersection

import os
import inspect
import gc
import re

from string import Template

PARENT_NODE_TEMPLATE = "{ data: { id: '$ID' }, classes: 'parent' },\n"
NODE_TEMPLATE = "{ data: { id: '$ID', label: '$LABEL' }, classes: '$CLASSES' },\n"
CHILD_NODE_TEMPLATE = "{ data: { id: '$ID', parent: '$PARENT', label: '$LABEL' }, classes: '$CLASSES' },\n"
EDGE_TEMPLATE = "{ data: { id: '$ID', source: '$SOURCE', target: '$TARGET' }, classes: '$CLASSES' },\n"
GRAPHDATA_TEMPLATE = "var elementsData = \n\n {nodes: [$NODES], \n\n edges: [$EDGES]}"


VIS = os.path.dirname(os.path.abspath(__file__))


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

        self.static_pipeline_json = os.path.join(VIS, "static", "static_pipeline.json")
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
                        if (
                            "concreteImplementations"
                            not in self.output_dict[pretty_abst_task]
                        ):
                            self.output_dict[pretty_abst_task][
                                "concreteImplementations"
                            ] = []

                        self.output_dict[pretty_abst_task][
                            "concreteImplementations"
                        ] += [pretty_task_name]
                        self.concrete_to_abst_mapper[
                            pretty_task_name
                        ] = pretty_abst_task
                else:
                    self.output_dict[pretty_task_name] = {"abstract": False}

    def add_dependencies_and_indexed_tasks(self):
        from cls_luigi.inhabitation_task import RepoMeta
        for k, v in self.repository.items():
            if not isinstance(k, RepoMeta.ClassIndex):
                if k.cls.abstract is not None:
                    if isinstance(v, Intersection):
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
                            self.output_dict[pretty_task]["configIndexes"][index] = [
                                pretty_indexed_t_name
                            ]

                            if pretty_indexed_t_name in self.concrete_to_abst_mapper:
                                pretty_indexed_t_name = self.concrete_to_abst_mapper[
                                    pretty_indexed_t_name
                                ]

                            if "inputQueue" not in self.output_dict[pretty_task]:
                                self.output_dict[pretty_task]["inputQueue"] = []

                            self.output_dict[pretty_task]["inputQueue"] += [
                                pretty_indexed_t_name
                            ]

                    if not isinstance(v, Constructor):
                        pretty_task = self._prettify_name(k.name)

                        predecessors = v.path
                        if predecessors:
                            for p in predecessors[0]:
                                pretty_p = self._prettify_name(p.name.cls_tpe)
                                if pretty_p in self.concrete_to_abst_mapper:
                                    pretty_p = self.concrete_to_abst_mapper[pretty_p]
                                if pretty_task in self.concrete_to_abst_mapper:
                                    pretty_task = self.concrete_to_abst_mapper[
                                        pretty_task
                                    ]
                                if "inputQueue" not in self.output_dict[pretty_task]:
                                    self.output_dict[pretty_task]["inputQueue"] = []
                                self.output_dict[pretty_task]["inputQueue"] += [
                                    pretty_p
                                ]

    def dump_static_repo_json(self):
        outfile_name = self.static_pipeline_json
        dump_json(outfile_name, self.output_dict)


class StaticVisualization:
    """
    Creates a cytoscape representation of all relevant pipeline components.

    usage:
      StaticVisualization.createGraph()
    """

    FILTER_PACKAGES = ["luigi", "cls_luigi"]

    def __init__(self):
        from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator
        self.repo_meta = RepoMeta
        self.repository = self.repo_meta.repository

        self.graphdata_js = os.path.join(VIS, "static", "graphdata.js")
        if os.path.exists(self.graphdata_js):
            os.remove(self.graphdata_js)

        self.repo_tasks = self._get_classes_with_multiple_inheritance(
            Task, LuigiCombinator
        )
        self.pure_tasks = self._get_classes_with_multiple_inheritance(
            Task, exclude_classes=[LuigiCombinator]
        )
        self.all_tasks = self.repo_tasks + self.pure_tasks

    @classmethod
    def add_package_to_filter(cls, package):
        cls.FILTER_PACKAGES.append(package)

    def _filter_classes(self, classes_to_filter=[]):
        for cls in classes_to_filter:
            if cls in self.repo_tasks:
                self.repo_tasks.remove(cls)
            if cls in self.pure_tasks:
                self.pure_tasks.remove(cls)
        self.all_tasks = self.repo_tasks + self.pure_tasks

    def create_graph(self):
        """
        Generates the 'graphdata.js' file that is used to visualize the overall pipeline structure.

        """
        graph_template = Template(GRAPHDATA_TEMPLATE)

        # function to get all nodes as filled string template
        # replace $NODES in graph_template

        nodes = set()
        edges = set()
        # Maps from already added parent id to a set of nodes that ist should contain
        parent_nodes = {}

        for cls in self.all_tasks:
            # Add node and edges for pure tasks.
            if cls in self.pure_tasks:
                node = Template(NODE_TEMPLATE)
                data = {
                    "ID": cls.__name__,
                    "LABEL": cls.__name__,
                    "CLASSES": "luigi-task outline",
                }
                nodes.add(node.substitute(data))
                if "requires" in cls.__dict__:
                    self._analyse_requires_and_add_edges(edges, cls)
                self.pure_tasks.remove(cls)
                break

            # Else we deal with LuigiCombinator Tasks
            # inherited_classes = [x for x in list(inspect.getmro(cls)[1:]) if x in self.all_tasks]

            class_chain = self.repo_meta._get_class_chain(cls)
            current_class = class_chain[0]
            filtered_upstream = [x for x in class_chain[1] if x in self.all_tasks]
            filtered_downstream = [x for x in class_chain[2] if x in self.all_tasks]
            result_set_for_parent = frozenset()

            if "requires" in cls.__dict__:
                self._analyse_requires_and_add_edges(edges, cls)

            # only inhabitation chains from abstract classes should be visulized using parent/compound nodes
            if cls.abstract:
                new_parent = True
                for key, set_of_classes in parent_nodes.items():
                    if cls in set_of_classes:
                        new_set = (
                            set_of_classes.union(frozenset(filtered_upstream))
                            .union(frozenset(filtered_downstream))
                            .union(frozenset([current_class]))
                        )
                        parent_nodes[key] = new_set
                        new_parent = False
                        break

                if new_parent:
                    result_set_for_parent = (
                        result_set_for_parent.union(frozenset(filtered_upstream))
                        .union(frozenset(filtered_downstream))
                        .union(frozenset([current_class]))
                    )
                    new_id = "PARENT_" + str(id(cls))
                    parent_nodes[new_id] = result_set_for_parent

            # directly add inheritance edge if there should be any.
            if filtered_upstream:
                inherited_classes = cls.__bases__
                filtered_inherited_classes = [
                    item for item in inherited_classes if item in filtered_upstream
                ]
                for direct_parent in filtered_inherited_classes:
                    edge = Template(EDGE_TEMPLATE)
                    data = {
                        "ID": direct_parent.__name__ + "_" + cls.__name__,
                        "SOURCE": direct_parent.__name__,
                        "TARGET": cls.__name__,
                        "CLASSES": "inheritance",
                    }
                    edges.add(edge.substitute(data))

        for key, set_of_classes in parent_nodes.items():
            node = Template(PARENT_NODE_TEMPLATE)
            data = {"ID": key}
            nodes.add(node.substitute(data))

            edge = Template(EDGE_TEMPLATE)
            data = {"ID": key, "SOURCE": key, "TARGET": key, "CLASSES": "loop"}
            edges.add(edge.substitute(data))

            for cls in set_of_classes:
                node = Template(CHILD_NODE_TEMPLATE)
                data = {
                    "ID": cls.__name__,
                    "LABEL": cls.__name__,
                    "PARENT": str(key),
                    "CLASSES": (
                        "config-domain-task"
                        if issubclass(cls, WrapperTask)
                        else "abstract-task"
                        if cls.abstract
                        else "concrete-task"
                    )
                    + " outline",
                }
                nodes.add(node.substitute(data))

                self.repo_tasks.remove(cls)

        # for rest of not visulized task, draw them!
        for cls in self.pure_tasks + self.repo_tasks:
            node = Template(NODE_TEMPLATE)
            data = {
                "ID": cls.__name__,
                "LABEL": cls.__name__,
                "CLASSES": (
                    "luigi-task"
                    if cls in self.pure_tasks
                    else "config-domain-task"
                    if issubclass(cls, WrapperTask)
                    else "abstract-task"
                    if cls.abstract
                    else "concrete-task"
                )
                + " outline",
            }
            nodes.add(node.substitute(data))
            if "requires" in cls.__dict__:
                self._analyse_requires_and_add_edges(edges, cls)

        graph_template = graph_template.substitute(
            {
                "NODES": "".join([node for node in nodes]),
                "EDGES": "".join([edge for edge in edges]),
            }
        )

        with open(self.graphdata_js, "w") as fp:
            fp.write(graph_template)

    def _analyse_requires_and_add_edges(self, edges, cls):
        from cls_luigi.inhabitation_task import ClsParameter
        method_object = cls.requires
        source_code = inspect.getsource(method_object)
        return_code = self._remove_empty_characters(
            self._remove_quoted_text(
                self._remove_between_delimiters(source_code, '"""', '"""').split(
                    "return"
                )[-1]
            )
        )
        for task in self.all_tasks:
            pattern = r"(?:\W|^){}(?:\W|$)".format(re.escape(task.__name__))

            if bool(re.search(pattern, return_code)):
                edge = Template(EDGE_TEMPLATE)
                data = {
                    "ID": task.__name__ + "_" + cls.__name__,
                    "SOURCE": task.__name__,
                    "TARGET": cls.__name__,
                    "CLASSES": "",
                }
                edges.add(edge.substitute(data))

                # uses reflection and __dict__ to see if class has ClsParameters defined,
                # that are defined in class and not by parent.
        cls_parameters = [
            name
            for name, param in inspect.getmembers(cls)
            if not name.startswith("__")
            and isinstance(param, ClsParameter)
            and name in cls.__dict__
        ]

        for param in cls_parameters:
            if "self." + param + "()" in return_code:
                parameter_object = getattr(cls, param)
                parameter_object_dict = parameter_object.tpe
                if not isinstance(parameter_object_dict, dict):
                    parameter_object_dict = {1: parameter_object_dict}

                for return_type in parameter_object_dict.values():
                    id_name = return_type.name.cls_tpe.split(".")[-1]
                    edge = Template(EDGE_TEMPLATE)
                    data = {
                        "ID": id_name + "_" + cls.__name__,
                        "SOURCE": id_name,
                        "TARGET": cls.__name__,
                        "CLASSES": "",
                    }
                    edges.add(edge.substitute(data))

    def _get_classes_with_multiple_inheritance(
        self, *classes, exclude_classes=[], exclude_packages=[]
    ):
        for packs in exclude_packages:
            StaticVisualization.add_package_to_filter(packs)
        all_classes = []
        for loaded_class in self._get_loaded_classes():
            if (
                inspect.isclass(loaded_class)
                and all(issubclass(loaded_class, cls) for cls in classes)
                and not any(issubclass(loaded_class, cls) for cls in exclude_classes)
            ):
                add_to_list = True

                for package_name in self.FILTER_PACKAGES:
                    if loaded_class.__module__.startswith(package_name):
                        add_to_list = False
                        break

                if add_to_list:
                    all_classes.append(loaded_class)

        return all_classes

    @staticmethod
    def _remove_empty_characters(input_string):
        return re.sub(r"\s+", "", input_string)

    @staticmethod
    def _remove_quoted_text(input_string):
        pattern = r"'[^']*'|\"[^\"]*\""
        return re.sub(pattern, "", input_string)

    @staticmethod
    def _remove_between_delimiters(string, start_delim, end_delim):
        parts = string.split(start_delim)
        if len(parts) > 1:
            first_part = parts[0]
            last_part = parts[-1].split(end_delim, 1)[-1]
            return first_part + last_part
        else:
            return string

    @staticmethod
    def _get_loaded_classes():
        all_classes = []
        for obj in gc.get_objects():
            if inspect.isclass(obj):
                all_classes.append(obj)

        return all_classes
