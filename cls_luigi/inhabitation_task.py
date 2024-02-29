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

import functools
import importlib


from typing import Any, TypeVar, Generic, Union, List, Dict, Tuple, Set
from typing import Type as PyType
from cls.fcl import FiniteCombinatoryLogic, InhabitationResult
from cls.types import Type, Constructor, Arrow
import cls.cls_json as cls_json
from dataclasses import dataclass, field

import luigi
from luigi.task_register import Register
from multiprocessing import Process

from cls_luigi.repo_visualizer.json_io import load_json, dump_json
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

# CONFIG = "config.json"
#
#
# class TaskProcessingTime(object):
#
#     '''
#     A mixin that when added to a luigi task, will save
#     the tasks execution time to the dynamic_pipeline.json
#     '''
#     @luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
#     def save_execution_time_to_dynamic_pipeline_json(self, processing_time):
#         def _prettify_task_name(task):
#             listed_task_id = task.split("_")
#             return listed_task_id[0] + "_" + listed_task_id[-1]
#
#         task_pretty_name = _prettify_task_name(self.task_id)
#
#         try:
#             dynamic_pipeline_path = load_json(CONFIG)["dynamic_pipeline"]
#             dynamic_pipeline_json = load_json(dynamic_pipeline_path)
#
#             for pipeline_ix in dynamic_pipeline_json.keys():
#                 for task in dynamic_pipeline_json[pipeline_ix].keys():
#                     if task == task_pretty_name:
#                         dynamic_pipeline_json[pipeline_ix][task]["processingTime"] = processing_time
#                         break # we know that in each pipeline the tasks are unique, so once we find the task we are looking for we break the inner loop
#             dump_json(dynamic_pipeline_path, dynamic_pipeline_json)
#
#         except FileNotFoundError as e:
#             msg = 'There is either a problem with the path to "dynamic_pipeline" or you are running with local_scheduler = True'
#             print(msg)
#             return FileNotFoundError
#             # Daniel: maybe remove msg for less cluttered logs and remove return of exception since no one is handling it.
#
#




@dataclass
class TaskState(object):
    fcl: FiniteCombinatoryLogic = field(init=True)
    target: Type = field(init=True)
    result: InhabitationResult | None = field(init=False, default=None)
    position: int = field(init=False, default=-1)
    stopped: bool = field(init=False, default=False)
    processes: list[Process] = field(init=False, default_factory=lambda: [])
    worker_scheduler_factory: luigi.interface._WorkerSchedulerFactory = field(init=True,
                                                                              default_factory=luigi.interface._WorkerSchedulerFactory)


states: dict[str, TaskState] = dict()


# class InhabitationTask(luigi.Task, TaskProcessingTime):
class InhabitationTask(luigi.Task):

    accepts_messages = True

    def requires(self):
        return []

    def run(self):
        global states
        state = states[self.task_id]
        if state.result is None:
            state.result = state.fcl.inhabit(state.target)

        while True:
            if not self.scheduler_messages.empty():
                msg = self.scheduler_messages.get()
                content = msg.content
                if content == "stop":
                    msg.respond(f"Stopping: {self.task_id}")
                    for p in state.processes:
                        p.join()
                    state.stopped = True
                    break
                number = -1
                try:
                    number = int(content)
                except ValueError:
                    if content != "next":
                        msg.respond(f"Not understood: {content}")
                        continue
                state.position = number if number >= 0 else state.position + 1
                if not state.result.infinite and state.position >= state.result.size():
                    msg.respond(f"Inhabitant not present: {state.position}")
                    state.position = -1
                    continue
                next_task = state.result.evaluated[state.position]()
                try:
                    msg.respond(f"Running {state.position}")
                except BaseException as ex:
                    msg.respond(f"Error: {str(ex)}")
                    continue

                def p(next_task, factory):
                    luigi.build([next_task], worker_scheduler_factory=factory)

                task_process = Process(target=p, args=(next_task, state.worker_scheduler_factory))
                task_process.start()
                state.processes.append(task_process)
                # p(next_task, state.worker_scheduler_factory)

    def complete(self):
        return states[self.task_id].stopped


class InhabitationTest(luigi.Task):
    done = luigi.BoolParameter(default=False)
    id = luigi.IntParameter()

    accepts_messages = True

    def requires(self):
        return []

    def run(self):
        while True:
            if not self.scheduler_messages.empty():
                msg = self.scheduler_messages.get()
                break
        self.done = True

    def complete(self):
        return self.done


ConfigIndex = TypeVar("ConfigIndex")


class ClsParameter(luigi.Parameter, Generic[ConfigIndex]):

    @functools.cached_property
    def decoder(self):
        return CLSLuigiDecoder()

    @functools.cached_property
    def encoder(self):
        return CLSLugiEncoder()

    def __init__(self, tpe: Type | dict[ConfigIndex, Type], **kwargs):
        kwargs["positional"] = False
        super(ClsParameter, self).__init__(**kwargs)
        self.tpe = tpe

    def parse(self, serialized):
        return self.decoder.decode(serialized)

    def serialize(self, x):
        return self.encoder.encode(x)


class RepoMeta(Register):
    repository: dict[Any, Type] = {}
    subtypes: dict['RepoMeta.TaskCtor', set['RepoMeta.TaskCtor']] = {}

    @staticmethod
    def _get_all_upstream_classes(target: PyType) -> Tuple[PyType, List[PyType]]:
        """
        This method returns a tuple. The first item is the given target, while the second item is
        a list of all upstream classes.

        Parameters
        ----------
        target: PyType
            The class for which you want to know the abstract upper classes.

        Returns
        -------
        Tuple[PyType, List[PyType]]
            A tuple containing the target class and all upstream classes.
        """
        next_target = RepoMeta.subtypes.get(RepoMeta.TaskCtor(target))
        list_of_all_upstream_classes = []
        if next_target:
            for item in next_target:
                list_of_all_upstream_classes.extend(RepoMeta._get_list_of_all_upstream_classes(item.tpe))
        return (target, list_of_all_upstream_classes)

    @staticmethod
    def _get_list_of_all_upstream_classes(target: PyType) -> List[PyType]:
        """
        This method returns a list of all upstream classes till it its the empty set.
        Uses information of the subtypes dict to get full chain. The resulting list will
        include the target class itself as head.

        Parameters
        ----------
        target: PyType
            The class for which you want to know the abstract upper classes.

        Returns
        -------
        List[PyType]
            A list of the  classes found till it hit the top. The list is sorted according to first seen classes.
        """
        list_of_all_upstream_classes = []
        list_of_all_upstream_classes.append(target)
        next_target = RepoMeta.subtypes.get(RepoMeta.TaskCtor(target))
        if not next_target:
            return list_of_all_upstream_classes

        else:
            for item in next_target:
                    list_of_all_upstream_classes.extend(RepoMeta._get_list_of_all_upstream_classes(item.tpe))
                    return list_of_all_upstream_classes

    @staticmethod
    def _get_all_upstream_abstract_classes(target: PyType) -> Tuple[PyType, List[PyType]]:
        """
        This method returns a tuple which contains the target as the first element
        and a list of all found abstract classes as the second element.

        Parameters
        ----------
        target: PyType
            The class for which you want to know the abstract upper classes.

        Returns
        -------
            Tuple[PyType, List[PyType]]: a tuple containing the target class and all upstream abstract classes.
        """
        next_target = RepoMeta.subtypes.get(RepoMeta.TaskCtor(target))
        list_of_all_upstream_classes = []
        if next_target:
            for item in next_target:
                list_of_all_upstream_classes.extend(RepoMeta._get_list_of_all_upstream_classes(item.tpe))
        return (target, list_of_all_upstream_classes)

    @staticmethod
    def _get_list_of_all_upstream_abstract_classes(target: PyType) -> List[PyType]:
        """
        This method can be used to get all abstract classes that are reachable from a given target class.
        It uses the information of the subtypes dict to find all abstract classes on the way, till it uses
        a class as key for the dict and gets an empty set back. It is possible that the target class is
        included in the resulting list as head, if it is abstract itself.

        Parameters
        ----------
        target: PyType
            The class for which you want to know the abstract upper classes

        Returns
        -------
        List[PyType]
            A list of the abstract classes found till it hit the top. The list is sorted according to first seen classes.
            So the head of the list is the first seen abstract class.
        """
        list_of_abstract_parents = [] # sorted in the sense of first item in the list is first encountered abstract node.
        if target.abstract == True:
            list_of_abstract_parents.append(target)

        next_target = RepoMeta.subtypes.get(RepoMeta.TaskCtor(target))
        if not next_target:
            return list_of_abstract_parents

        else:
            for item in next_target:
                list_of_abstract_parents.extend(RepoMeta._get_list_of_all_upstream_abstract_classes(item.tpe))
                return list_of_abstract_parents

    @staticmethod
    def _get_all_downstream_classes(target: PyType) -> Tuple[PyType, Set[PyType]]:
        """
        Get the set of all downstream classes for a given target class.

        This method uses the `__get_set_of_all_downstream_classes` method to find all downstream
        classes for the given `target` class, and returns a tuple containing the `target` class and
        the set of downstream classes.

        Parameters
        ----------
        target: PyType
            The target class to find downstream classes for.

        Returns
        -------
        Tuple[PyType, Set[PyType]]
            A tuple containing the `target` class and the set of downstream classes for the given `target`.
        """
        return (target, RepoMeta.__get_set_of_all_downstream_classes([target]))

    @staticmethod
    def _get_all_downstream_abstract_classes(target: PyType) -> Tuple[PyType, Set[PyType]]:
        """
        Get all downstream abstract classes for a given class object.

        This method returns a tuple of the target class object and a set of all class objects that are
        downstream from the target and are abstract.

        Parameters
        ----------
        target: PyType
            The target class object for which the downstream abstract classes are to be returned.

        Returns
        -------
        Tuple[PyType, Set[PyType]]
            A tuple containing the target class object as the first element and a set of all
            class objects that are downstream from the target and are abstract.
        """
        result_set = set()
        all_downstream_classes = RepoMeta._get_all_downstream_classes(target)[1]
        for item in all_downstream_classes:
            if item.abstract:
                result_set.add(item)
        return (target, result_set)

    @staticmethod
    def _get_all_downstream_concrete_classes(target: PyType) -> Tuple[PyType, Set[PyType]]:
        """
        Get all downstream concrete classes for a given class object.

        This method returns a tuple of the target class object and a set of all class objects that are
        downstream from the target and are concrete.

        Parameters
        ----------
        target: PyType
            The target class object for which the downstream concrete classes are to be returned.

        Returns
        -------
        Tuple[PyType, Set[PyType]]
            A tuple containing the target class object as the first element and a set of all
            class objects that are downstream from the target and are concrete.
        """
        result_set = set()
        all_downstream_classes = RepoMeta._get_all_downstream_classes(target)[1]
        for item in all_downstream_classes:
            if not item.abstract:
                result_set.add(item)
        return (target, result_set)


    @staticmethod
    def __get_set_of_all_downstream_classes(targets: List[PyType], current_set: Set[PyType] = set()) -> Set[PyType]:
        """
        Get the set of all downstream classes for a given set of targets.

        This method uses a recursive approach to build up a set of downstream classes, starting
        with the `targets` and iteratively adding any downstream classes that are found.
        The `current_set` parameter is used to keep track of the set of downstream classes that
        have already been found, and is updated and returned after each recursive call.

        Parameters
        ----------
        targets: List[PyType]
            A list of target classes to find downstream classes for.
        current_set: Set[PyType]
            A set of classes that have already been found as downstream classes. This set is updated
            and returned after each recursive call.

        Returns
        -------
        Set[PyType]
            The set of all downstream classes for the given `targets`.
        """


        def helper_to_get_all_downstream_classes(target: PyType) -> Set[PyType]:
            """
            Get the set of downstream classes for a given target class.

            This method looks through the `subtypes` attribute of the class that this method is called
            on, and returns a set of classes that are downstream of the given `target` class.

            Parameters
            ----------
            target: PyType
                The target class to find downstream classes for.

            Returns
            -------
            Set[PyType]
                The set of downstream classes for the given `target` class.
            """
            list_of_downstream_classes = []
            for key, value in RepoMeta.subtypes.items():
                if not value:
                    continue
                else:
                    for task_ctor in value:
                        if target == task_ctor.tpe:

                            list_of_downstream_classes.append(key.tpe)
            return set(list_of_downstream_classes)

        set_of_downstream_classes = set()
        if len(targets) == 0:
            return current_set
        else:
            for target in targets:
                set_of_downstream_classes = set_of_downstream_classes.union(helper_to_get_all_downstream_classes(target))
            current_set = current_set.union(set_of_downstream_classes)
            return RepoMeta.__get_set_of_all_downstream_classes(list(set_of_downstream_classes), current_set)

    @staticmethod
    def _get_class_chain(target: PyType) ->  Tuple[PyType, Set[PyType], Set[PyType]]:
        """
        Get a tuple containing the target class and its upstream and downstream classes.

        This method uses the `_get_all_upstream_classes` and `_get_all_downstream_classes` methods to
        find the upstream and downstream classes for the given `target` class, and returns a tuple
        containing the `target` class, the set of upstream classes, and the set of downstream classes.

        Parameters
        ----------
        target: PyType
            The target class to find upstream and downstream classes for.

        Returns
        -------
        Tuple[PyType, Set[PyType], Set[PyType]]
            A tuple containing the `target` class, the set of upstream classes, and the set of downstream classes.
        """
        return (target, RepoMeta._get_all_upstream_classes(target)[1], RepoMeta._get_all_downstream_classes(target)[1])

    @staticmethod
    def _get_abstract_class_chain(target: PyType) -> Tuple[PyType, Set[PyType], Set[PyType]]:
        """
        Get a tuple containing the target class and its upstream and downstream abstract classes.

        This method uses the `_get_all_upstream_abstract_classes` and `_get_all_downstream_abstract_classes` methods
        to find the upstream and downstream abstract classes for the given `target` class, and returns a tuple
        containing the `target` class, the set of upstream abstract classes, and the set of downstream abstract classes.

        Parameters
        ----------
        target: PyType
            The target class to find upstream and downstream abstract classes for.

        Returns
        -------
        Tuple[PyType, Set[PyType], Set[PyType]]
            A tuple containing the `target` class, the set of upstream abstract classes, and the set of downstream abstract classes.
        """
        return (target, RepoMeta._get_all_upstream_abstract_classes(target)[1], RepoMeta._get_all_downstream_abstract_classes(target)[1])

    @staticmethod
    def _get_maximal_shared_upper_classes(targets: List[PyType]) -> Tuple[List[PyType], bool]:
        """
        Finds the maximal shared upper classes of the given targets.

        Parameters
        ----------
        targets : List[PyType]
            List of classes or tuples of classes to find maximal shared upper classes for.

        Returns
        -------
        Tuple[List[PyType], bool]
            A tuple containing the list of maximal shared upper classes and a boolean
            that indicates whether or not all upper classes are equal.
        """
        all_equal = False
        if len(targets) <= 1:
            return ([], True)
        lists = []
        for target in targets:
            if isinstance(target, tuple):
                lists.append(RepoMeta._get_list_of_all_upstream_classes(target[0])[:0:-1])
            else:
                lists.append(RepoMeta._get_list_of_all_upstream_classes(target)[:0:-1])

        if all(x == lists[0] for x in lists):
            all_equal = True

        prefix = [x[0] for x in zip(*lists) if all(x[0] == y for y in x[1:])]
        return (prefix, all_equal)

    @staticmethod
    def _delete_related_combinators(targets: List[PyType], repository: Dict[Any, Type] = repository) -> Dict[Any, Type]:
        """
        Removes the related combinators from the given repository based on the given targets
        and returns a copy of the new repository.

        Parameters
        ----------
        targets: List[PyType]
            List of classes or tuples of classes that the related combinators will be removed from.
        repository: Dict[Any, Type]
            Dictionary containing the combinators that will be checked and removed.

        Returns
        -------
        Dict[Any, Type]
            A copy of the original repository with the related combinators removed.
        """
        to_remove = []
        result_repository = repository.copy()
        for combinator in repository:
            if not isinstance(combinator, RepoMeta.ClassIndex):

                if issubclass(combinator.cls, tuple(targets)):
                    to_remove.append(combinator)

        for combinator in to_remove:
            result_repository.pop(combinator)
        return result_repository


    @staticmethod
    def filtered_repository(targets: List[PyType] = [], repository: Dict[Any, Type] = repository) -> Dict[Type, Any]:
        """
        Filters the repository to include only combinators related to the specified targets.

        This method filters the repository to include only combinators that are related to the specified targets.
        The targets can be either abstract classes or concrete classes.

        Parameters
        ----------
        targets: List[Type]
            A list of class objects for which related combinators should be included.
        repository: Dict[Type, Any]
            A dictionary containing the combinators in the repository.

        Returns
        -------
        Dict[Type, Any]
            A copy of the repository with only the related combinators included.
        """

        #setup datastructures to hold information on what to delete and want to keep
        global_combinators_to_delete = set()
        global_combinators_to_keep = set()

        #check if there are general upper classes that every target is sharing
        shared_upper_classes, all_equal = RepoMeta._get_maximal_shared_upper_classes(targets)

        # check if all targets share all upper classes, in that case we remove
        # the last class in the chain, since we want to remove combinators that
        # lay on the same level.
        if all_equal:
            if len(shared_upper_classes) > 1:
                shared_upper_classes.pop(-1)

        for target in targets:
            selected_classes = []
            combinators_to_delete = set()
            combinators_to_keep = set()

            if isinstance(target, tuple):
                # if tuple split tuple into the current target and the selected classes for it
                for class_object in target[1]:
                    if not class_object.abstract:
                        selected_classes.append(class_object)
                    else:
                        for item in RepoMeta._get_all_downstream_concrete_classes(class_object):
                            selected_classes.append(item)
                target = target[0]

            if target.abstract:
                if selected_classes:
                    # if there are selected classes for the target, only add them to keep and
                    # all the remaining one to delete.
                    combinators_to_keep.update([x for x in RepoMeta._get_all_downstream_concrete_classes(target)[1] if x in selected_classes])
                    combinators_to_delete.update([x for x in RepoMeta._get_all_downstream_concrete_classes(target)[1] if x not in selected_classes])
                else:
                    # if no selected classes present, just add all concrete classes downstream to keep.
                    combinators_to_keep.update([x for x in RepoMeta._get_all_downstream_concrete_classes(target)[1]])

            else:
                # if target is concrete, just add it to keep.
                combinators_to_keep.add(target)

            # go up to the next abstract class, if there is a concrete class on the way,
            # add the corresponding combinator to keep
            # From the new abstract class, search downstream and add corresponding combinators to delete
            for class_object in RepoMeta._get_all_upstream_classes(target)[1]:
                if class_object in shared_upper_classes:
                    break
                if not class_object.abstract:
                    combinators_to_keep.add(class_object)
                else:
                    combinators_to_delete.update(RepoMeta._get_all_downstream_concrete_classes(class_object)[1])
                    break
            global_combinators_to_keep.update(combinators_to_keep)
            global_combinators_to_delete.update(combinators_to_delete - combinators_to_keep)

        final_combinators_to_delete = global_combinators_to_delete - global_combinators_to_keep
        result_repository = RepoMeta._delete_related_combinators(targets= final_combinators_to_delete, repository= repository)


        return result_repository

    @staticmethod
    def get_list_of_variated_abstract_tasks(repository= repository, subtypes= subtypes) -> List[Type]:
        """
        Get a list of variated abstract tasks.

        This method returns a list of abstract tasks that have been really been variated and thus
        should most likely be unique in any pipeline.

        Parameters
        ----------
        repository: Dict[Type, Any]
            A dictionary containing the combinators in the repository.
        subtypes: Dict[Type, List[Type]]
            A dictionary containing the subtypes of each class object.

        Returns
        -------
        List[Type]
            A list of class objects representing the variated abstract tasks.
        """
        abstract_tasks = set()
        subtypes_set = set()
        repo_set = set()

        for item in subtypes.keys():
            to_comp = None
            if isinstance(item, RepoMeta.WrappedTask):
                to_comp = item.cls
            else:
                to_comp = item.tpe
            subtypes_set.add(to_comp)

        for item in repository.keys():
            to_comp = None
            if isinstance(item, RepoMeta.WrappedTask):
                to_comp = item.cls
            else:
                to_comp = item.tpe
            repo_set.add(to_comp)


        abstract_tasks = subtypes_set - repo_set

        variated_abstract_tasks = set()
        for item in abstract_tasks:
            for repo in [str(x) for x in repository.values()]:
                if str(RepoMeta.cls_tpe(item)) in repo:
                    variated_abstract_tasks.add(item)

        return list(variated_abstract_tasks)

    @staticmethod
    def get_unique_abstract_task_validator() -> UniqueTaskPipelineValidator:
        """
        Returns an instance of the `UniqueTaskPipelineValidator` class with the list of variated abstract tasks as its argument.

        Returns
        -------
        UniqueTaskPipelineValidator
            instance of the `UniqueTaskPipelineValidator` class.
        """
        return UniqueTaskPipelineValidator(RepoMeta.get_list_of_variated_abstract_tasks())

    @staticmethod
    def cls_tpe(cls) -> str:
        return f'{cls.__module__}.{cls.__qualname__}'

    @dataclass(frozen=True)
    class TaskCtor(object):
        tpe: PyType[Any] = field(init=True, compare=False)
        cls_tpe: str = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "cls_tpe", RepoMeta.cls_tpe(self.tpe))

        def __str__(self):
            return self.cls_tpe

    @dataclass(frozen=True)
    class ClassIndex(Generic[ConfigIndex]):
        tpe: PyType[Any] = field(init=True, compare=False)
        at_index: ConfigIndex = field(init=True)
        cls_tpe: str = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "cls_tpe", RepoMeta.cls_tpe(self.tpe))

        def __str__(self):
            return f"{self.cls_tpe}_{str(self.at_index)}"

    @dataclass(frozen=True)
    class WrappedTask(object):
        cls: PyType[Any] = field(init=True, compare=False)
        has_index: bool = field(init=True, compare=False)
        cls_params: tuple[Tuple[str, ClsParameter[Any]]] = field(init=True)
        reverse_arguments: tuple[Any] = field(init=True)
        name: str = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "name", RepoMeta.cls_tpe(self.cls))

        def __call__(self, *args, **kwargs) -> Union['RepoMeta.WrappedTask', 'LuigiCombinator[Any]']:
            expected = len(self.cls_params) + 1 if self.has_index else len(self.cls_params)
            if expected == len(self.reverse_arguments):
                cls_param_names = (name for name, _ in self.cls_params)
                all_params = ("config_index", *cls_param_names) if self.has_index else cls_param_names
                arg_dict = dict(zip(all_params, reversed(self.reverse_arguments)))
                return super(RepoMeta, self.cls).__call__(*args, **(kwargs | arg_dict))
            else:
                arg = args[0].at_index if self.has_index and not self.reverse_arguments else args[0]
                return RepoMeta.WrappedTask(self.cls, self.has_index, self.cls_params, (arg, *self.reverse_arguments))

        def __str__(self):
            return self.name

    def __init__(cls, name, bases, dct):
        super(RepoMeta, cls).__init__(name, bases, dct)
        # Make sure to skip LuigiCombinator base class
        if cls.__module__ == RepoMeta.__module__ and cls.__qualname__ == "LuigiCombinator":
            return

        # Update subtype information
        cls_tpe: str = RepoMeta.cls_tpe(cls)
        RepoMeta.subtypes[RepoMeta.TaskCtor(cls)] = \
            {RepoMeta.TaskCtor(b)
             for b in bases
             if issubclass(b, LuigiCombinator) and not issubclass(LuigiCombinator, b)}

        # Update repository
        cls_params = [(name, param) for name, param in cls.get_params() if isinstance(param, ClsParameter)]
        index_set = RepoMeta._index_set(cls_tpe, cls_params)
        if not cls.config_domain is None:
            if not index_set:
                index_set = cls.config_domain
            else:
                index_set.intersection_update(cls.config_domain)
        combinator = RepoMeta.WrappedTask(cls, bool(index_set), tuple(cls_params), ())
        tpe = RepoMeta._combinator_tpe(cls, index_set, cls_params)
        if not cls.abstract:
            RepoMeta.repository[combinator] = tpe

        # Insert index combinators
        for idx in index_set:
            if not cls.abstract:
                RepoMeta.repository[RepoMeta.ClassIndex(cls, idx)] = Constructor(RepoMeta.ClassIndex(cls, idx))
            if not RepoMeta.ClassIndex(cls, idx) in RepoMeta.subtypes:
                RepoMeta.subtypes[RepoMeta.ClassIndex(cls, idx)] = set()
            for b in RepoMeta.subtypes[RepoMeta.TaskCtor(cls)]:
                if RepoMeta.ClassIndex(b.tpe, idx) in RepoMeta.subtypes:
                    RepoMeta.subtypes[RepoMeta.ClassIndex(cls, idx)].add(RepoMeta.ClassIndex(b.tpe, idx))

    @staticmethod
    def _combinator_tpe(cls: PyType[Any], index_set: set[Any], cls_params: list[Tuple[str, ClsParameter[Any]]]) -> Type:
        reverse_params = list(reversed(cls_params))
        if not index_set:
            tpe: Type = cls.return_type()
            for _, param in reverse_params:
                tpe = Arrow(param.tpe, tpe)
            return tpe
        else:
            def at_index(idx) -> Type:
                tpe: Type = cls.return_type(idx)
                for _, param in reverse_params:
                    if isinstance(param.tpe, Type):
                        tpe = Arrow(param.tpe, tpe)
                    else:
                        tpe = Arrow(param.tpe[idx], tpe)
                return Arrow(Constructor(RepoMeta.ClassIndex(cls, idx)), tpe)

            return Type.intersect(list(map(at_index, index_set)))

    @staticmethod
    def _index_set(cls_tpe: str, cls_params: list[Tuple[str, ClsParameter[Any]]]) -> set[Any]:
        index_set: set[Any] = set()
        for name, param in cls_params:
            if not index_set and isinstance(param.tpe, dict):
                index_set.update(param.tpe.keys())
                if not index_set:
                    raise IndexError(f"Error in parameter {cls_tpe}.{name}: cannot have empty index set")
            elif index_set and isinstance(param.tpe, dict):
                index_set.intersection_update(param.tpe.keys())
                if not index_set:
                    raise IndexError(f"Error in parameter {cls_tpe}.{name}: no index shared with all prior parameters")
        return index_set


class CLSLugiEncoder(cls_json.CLSEncoder):
    def __init__(self, **kwargs):
        super(CLSLugiEncoder, self).__init__(**kwargs)

    @staticmethod
    def _serialize_config_index(idx: RepoMeta.ClassIndex):
        return {"__type__": RepoMeta.cls_tpe(RepoMeta.ClassIndex),
                "module": idx.tpe.__module__,
                "task_class": idx.tpe.__qualname__,
                "index": idx.tpe.config_index.serialize(idx.at_index)}

    @staticmethod
    def _serialize_combinator(c: RepoMeta.WrappedTask):
        serialized_args = []
        args = list(c.reverse_arguments)
        if c.has_index and args:
            serialized_args.append({"config_index": c.cls.config_index.serialize(args.pop())})
        params = list(reversed(c.cls_params))
        while args:
            name, _ = params.pop()
            arg = args.pop()
            serialized_args.append({name: CLSLugiEncoder._serialize_combinator(arg)})

        return {"__type__": RepoMeta.cls_tpe(RepoMeta.WrappedTask),
                "module": c.cls.__module__,
                "task_class": c.cls.__qualname__,
                "arguments": serialized_args}

    @staticmethod
    def _serialize_task_ctor(ctor: RepoMeta.TaskCtor):
        return {"__type__": RepoMeta.cls_tpe(RepoMeta.TaskCtor),
                "module": ctor.tpe.__module__,
                "task_class": ctor.tpe.__qualname__}

    def combinator_hook(self, o):
        if isinstance(o, RepoMeta.WrappedTask):
            return CLSLugiEncoder._serialize_combinator(o)
        elif isinstance(o, RepoMeta.ClassIndex):
            return CLSLugiEncoder._serialize_config_index(o)
        else:
            return cls_json.CLSEncoder.combinator_hook(self, o)

    def constructor_hook(self, o):
        if isinstance(o, RepoMeta.TaskCtor):
            return CLSLugiEncoder._serialize_task_ctor(o)
        elif isinstance(o, RepoMeta.ClassIndex):
            return CLSLugiEncoder._serialize_config_index(o)
        else:
            return cls_json.CLSEncoder.constructor_hook(self, o)

    def default(self, o):
        if isinstance(o, RepoMeta.WrappedTask):
            return CLSLugiEncoder._serialize_combinator(o)
        else:
            return super(CLSLugiEncoder, self).default(o)


class CLSLuigiDecoder(cls_json.CLSDecoder):
    def __init__(self, **kwargs):
        super(CLSLuigiDecoder, self).__init__(**kwargs)

    @staticmethod
    def _deserialize_config_index(dct):
        module = importlib.import_module(dct["module"])
        task_class = getattr(module, dct["task_class"])
        return RepoMeta.ClassIndex(task_class, task_class.config_index.parse(dct["index"]))

    @staticmethod
    def _deserialize_combinator(dct):
        module = importlib.import_module(dct["module"])
        task_class = getattr(module, dct["task_class"])
        wrapped_task = None
        for c in RepoMeta.repository.keys():
            if isinstance(c, RepoMeta.WrappedTask) and c.name == RepoMeta.cls_tpe(task_class):
                wrapped_task = c
                break
        if not wrapped_task:
            raise RuntimeError(f"Cannot find WrappedTask for: {RepoMeta.cls_tpe(task_class)}")
        serialized_args = list(reversed(dct["arguments"]))
        if serialized_args and wrapped_task.has_index:
            arg = serialized_args.pop()
            wrapped_task = wrapped_task(wrapped_task.cls.config_index.parse(arg))
        while serialized_args:
            arg = list(serialized_args.pop().values())[0]
            wrapped_task = wrapped_task(arg)
        return wrapped_task

    @staticmethod
    def _deserialize_task_ctor(dct):
        module = importlib.import_module(dct["module"])
        task_class = getattr(module, dct["task_class"])
        return RepoMeta.TaskCtor(task_class)

    def combinator_hook(self, dct):
        if "__type__" in dct:
            tpe = dct["__type__"]
            if tpe == RepoMeta.cls_tpe(RepoMeta.ClassIndex):
                return CLSLuigiDecoder._deserialize_config_index(dct)
            elif tpe == RepoMeta.cls_tpe(RepoMeta.WrappedTask):
                return CLSLuigiDecoder._deserialize_combinator(dct)
        return cls_json.CLSDecoder.combinator_hook(self, dct)

    def constructor_hook(self, dct):
        if "__type__" in dct:
            tpe = dct["__type__"]
            if tpe == RepoMeta.cls_tpe(RepoMeta.ClassIndex):
                return CLSLuigiDecoder._deserialize_config_index(dct)
            elif tpe == RepoMeta.cls_tpe(RepoMeta.TaskCtor):
                return CLSLuigiDecoder._deserialize_task_ctor(dct)
        return cls_json.CLSDecoder.combinator_hook(self, dct)

    def __call__(self, dct):
        if "__type__" in dct:
            tpe = dct["__type__"]
            if tpe == RepoMeta.cls_tpe(RepoMeta.WrappedTask):
                return CLSLuigiDecoder._deserialize_combinator(dct)
        return super(CLSLuigiDecoder, self).__call__(dct)


class LuigiCombinator(Generic[ConfigIndex], metaclass=RepoMeta):
    config_index = luigi.OptionalParameter(positional=False, default="")
    config_domain: set[ConfigIndex] | None = None
    abstract: bool = False


    @classmethod
    def return_type(cls, idx: ConfigIndex = None) -> Type:
        if idx is None:
            return Constructor(RepoMeta.TaskCtor(cls))
        else:
            return Constructor(RepoMeta.TaskCtor(cls), Constructor(RepoMeta.ClassIndex(cls, idx)))

