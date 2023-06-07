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

"""Top-level package for cls-luigi."""

__version__ = "0.0.0"


import os

CURRENT_WORKING_DIRECTORY = os.getcwd()
INVISIBLE_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, '.cls_luigi')
HASH_PATH = os.path.join(INVISIBLE_PATH, 'hash_files')
RESULTS_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, 'results')
