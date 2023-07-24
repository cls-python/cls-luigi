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

import json
import os


def load_json(name):
    full_path = get_full_path(name)
    try:
        with open(full_path, 'r') as j:
            return json.load(j)
    except FileNotFoundError:
        print('File: "{}" not found'.format(full_path))
        raise FileNotFoundError


def dump_json(name, obj):
    full_path = get_full_path(name)
    with open(full_path, 'w+') as j:
        json.dump(obj, j, indent=6)


def get_full_path(name):
    return os.path.join(os.path.dirname(__file__), name)


if __name__ == "__main__":
    load_json('config.json')