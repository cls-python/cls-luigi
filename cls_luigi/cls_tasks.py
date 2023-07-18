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
from cls_luigi.util.util import get_unique_task_id


class ClsTask(luigi.Task, LuigiCombinator):
    """
    Abstract class representing a CLS-Luigi task. It combines the functionality of `luigi.Task` and`LuigiCombinator`.
    If RESULT_PATH directory doesn't exist, it will be created automatically.
    """
    abstract = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_output_dir()

    def get_unique_output_path(self, file_name: str, outputs_dir: str = RESULTS_PATH) -> str:
        """
        Returns a unique output path for the given file name using the task's unique id and the RESULT_PATH directory.

        :param file_name: String; the file name to be used.
        :param outputs_dir: String; the directory to be used.

        :output: String; the unique output path.

        """
        task_id = get_unique_task_id(task=self)

        return join(
            outputs_dir, f"{task_id}-{file_name}"
        )

    @staticmethod
    def create_output_dir(outputs_dir=RESULTS_PATH) -> None:
        if not exists(outputs_dir):
            os.mkdir(outputs_dir)
            print(f"Created output directory: {outputs_dir}")


class ClsWrapperTask(luigi.WrapperTask, LuigiCombinator):
    """
    Abstract class representing a CLS-Luigi wrapper task. It combines the functionality of `luigi.WrapperTask` and `LuigiCombinator`.
    """
    abstract = True


class ClsExternalTask(luigi.ExternalTask, LuigiCombinator):
    """
    Abstract class representing a CLS-Luigi external task. It combines the functionality of `luigi.ExternalTask` and `LuigiCombinator`.
    """
    abstract = True
