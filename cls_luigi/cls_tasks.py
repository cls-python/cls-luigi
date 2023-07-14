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

"""
This file contains the CLSBaseTask implementation, which is used to extend the actual implementation of
luigi.Task, luigi.WrapperTask and luigi.ExternalTask with helper methods to make the usage of our
framework a bit easier.
"""
import os
import random

from cls_luigi.inhabitation_task import LuigiCombinator
from cls_luigi import INVISIBLE_PATH, HASH_PATH, RESULTS_PATH
import hashlib
import luigi
from pathlib import Path
from os.path import join, exists
from os import makedirs


class ClsBaseTask():
    initialized = False

    def __init__(self) -> None:
        if not ClsBaseTask.initialized:

            if not exists(INVISIBLE_PATH):
                makedirs(INVISIBLE_PATH)

            if not exists(HASH_PATH):
                makedirs(HASH_PATH)

            if not exists(RESULTS_PATH):
                makedirs(RESULTS_PATH)

            ClsBaseTask.initialized = True

        self.result_path = RESULTS_PATH
        self.hash_path = HASH_PATH

    def get_variant_filename(self, name) -> str:

        md5_hash = self.get_md5_hexdigest(self)
        label = "{0}-#{1}#".format(self.__class__.__name__, md5_hash)
        hash_label_file_path = Path(join(self.hash_path, label))

        deps_tree = self.get_deps_tree()
        if hash_label_file_path.exists() is False:
            with hash_label_file_path.open(mode='w+') as hash_file:
                hash_file.write(deps_tree)

        variant_path_and_name = join(self.result_path, label)
        variant_path_and_name += "-" + name

        return variant_path_and_name

    @staticmethod
    def get_md5_hexdigest(task: luigi.Task) -> str:
        hash_value = hashlib.md5()
        task = str(task)
        hash_value.update(task.encode())
        return hash_value.hexdigest()

    def get_deps_tree(self, task=None, indent='', last=True):
        if task is None:
            task = self
            md5_hash = ""
        else:
            md5_hash = self.get_md5_hexdigest(task)

        name = task.__class__.__name__
        result = '\n' + indent
        if last is True:
            result += '└─--'
            indent += '   '
        else:
            result += '|--'
            indent += '|  '

        result += '[{0}-#{1}#]'.format(name, md5_hash)
        children = luigi.task.flatten(task.requires())
        for index, child in enumerate(children):
            result += self.get_deps_tree(child, indent, (index + 1) == len(children))

        return result


# class ClsBaseTask():
#     """
#     This class is used to implement some helper methods that all tasks in a CLS-Luigi pipeline should have.
#
#     The key methods are:
#
#     * :py:meth:`get_variant_filename` - This method creates a unique filename for (intermediate) results, which remains unique across the executed pipelines. For this purpose a hash value is created and persisted to disk, which can be looked up afterwards to trace the executed tasks back to this file.
#     """
#
#     initialized = False
#
#     def __init__(self) -> None:
#         if not ClsBaseTask.initialized:
#
#             if not exists(INVISIBLE_PATH):
#                 makedirs(INVISIBLE_PATH)
#
#             if not exists(HASH_PATH):
#                 makedirs(HASH_PATH)
#
#             if not exists(RESULTS_PATH):
#                 makedirs(RESULTS_PATH)
#
#             ClsBaseTask.initialized = True
#
#         self.result_path = RESULTS_PATH
#         self.hash_path = HASH_PATH
#
#
#     def __get_flatten_data(self, data : luigi.target.FileSystemTarget|tuple|dict) -> list:
#         """
#         Helper method to flatten different possible input data types into a flattened list.
#
#         :param data: the data structure that should be flattened to a list.
#         :type data: luigi.target.FileSystemTarget | tuple | dict
#         :return: a flattened list of the data provided.
#         :rtype: list
#         """
#         flattened_data = []
#         if isinstance(data, luigi.target.FileSystemTarget):
#             flattened_data.append(data)
#
#         elif isinstance(data, (list, tuple)):
#             for item in data:
#                 flattened_data.extend(self.__get_flatten_data(item))
#         elif isinstance(data, dict):
#             for _, value in data.items():
#                 flattened_data.extend(self.__get_flatten_data(value))
#         else:
#             # error case ? just throw away?
#             pass
#
#         return flattened_data
#
#     def get_variant_filename(self, name="") -> str:
#         """
#         Returns a variant filename based on the provided name.
#         Also does hashing of the name, since is has to be unique over every pipeline.
#
#         :param name: optional name for the variant, defaults to ""
#         :type name: str, optional
#         :return: the variant filename.
#         :rtype: str
#         """
#         if name == "":
#             name = self.__class__.__name__ + "_result"
#         #makedirs(dirname(self.hash_path), exist_ok=True)
#         hash_value = hashlib.md5()
#         label = ""
#
#         if isinstance(self.input(), luigi.target.FileSystemTarget):
#             input_file = Path(self.input().path)
#             label = self.__helper_variant_label(input_file)
#
#         elif isinstance(self.input(), (list, tuple, dict)):
#
#             flattened_data = self.__get_flatten_data(self.input())
#             var_label_name = []
#             for item in flattened_data:
#                 input_file = Path(item.path)
#                 var_label_name.append(self.__helper_variant_label(input_file))
#             label = "(" + (", ".join(var_label_name)) + ")" + " --> " + \
#                 self.__class__.__name__ if len(
#                     ", ".join(var_label_name)) > 0 else self.__class__.__name__
#
#         else:
#             label = self.__class__.__name__
#
#         label = "(" + label + "_" + self.__class__.__name__ + "-" +  name + ")"
#
#         hash_value.update(label.encode())
#         path = Path(join(self.hash_path, hash_value.hexdigest()))
#         if not path.is_file():
#             with path.open(mode='w+') as hash_file:
#                 hash_file.write(label)
#         return self.__class__.__name__ + "_" + "#" + hash_value.hexdigest() + "#" + "_" + name if label else self.__class__.__name__ + "_" + name
#
#     def __helper_variant_label(self, input_file):
#
#         input_filename = input_file.name
#         try:
#             _, lookup_hash, _ = input_filename.split("#", maxsplit=2)
#             if len(lookup_hash) == 32:
#                 hash_file = Path(join(self.hash_path,  lookup_hash))
#                 if hash_file.is_file():
#                     with hash_file.open(mode='r') as f:
#                         replacement_of_hash = f.read()
#                         # label = "(" + replacement_of_hash + ")" + " --> " + self.__class__.__name__ if len(
#                         #     input_filename) > 0 else self.__class__.__name__
#                         label = replacement_of_hash
#                         return label
#                 else:
#                     raise ValueError
#             else:
#                 raise ValueError
#
#         except ValueError:
#             label = input_filename
#             return label
#
#     def create_result_file(self, file_name):
#         return luigi.LocalTarget(join(self.result_path, self.get_variant_filename(file_name)))


class ClsTask(luigi.Task, LuigiCombinator, ClsBaseTask):
    """
    Abstract class representing a CLS-Luigi task. It combines the functionality of `luigi.Task`, `LuigiCombinator`, and `ClsBaseTask`.
    """
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ClsBaseTask.__init__(self)


class ClsWrapperTask(luigi.WrapperTask, LuigiCombinator, ClsBaseTask):
    """
    Abstract class representing a CLS-Luigi wrapper task. It combines the functionality of `luigi.WrapperTask`, `LuigiCombinator`, and `ClsBaseTask`.
    """
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ClsBaseTask.__init__(self)


class ClsExternalTask(luigi.ExternalTask, LuigiCombinator, ClsBaseTask):
    """
    Abstract class representing a CLS-Luigi external task. It combines the functionality of `luigi.ExternalTask`, `LuigiCombinator`, and `ClsBaseTask`.
    """
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ClsBaseTask.__init__(self)
