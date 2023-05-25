import sys
sys.path.append('../')

from inhabitation_task import LuigiCombinator
import hashlib
import luigi
from collections.abc import Iterable
from pathlib import Path
from os.path import dirname
from os import makedirs
from os.path import join as pjoin



class CLSBaseTask():
    global_hash_path = luigi.Parameter(default="hash_files/")

    def __get_flatten_data(self, data):
        flattened_data = []
        if isinstance(data, luigi.target.FileSystemTarget):
            flattened_data.append(data)

        elif isinstance(data, (list, tuple)):
            for item in data:
                flattened_data.extend(self.__get_flatten_data(item))
        elif isinstance(data, dict):
            for _, value in data.items():
                flattened_data.extend(self.__get_flatten_data(value))
        else:
            # error case ? just throw away?
            pass

        return flattened_data

    def _get_variant_filename(self, name=""):
        if name == "":
            name = self.__class__.__name__ + "_result"
        makedirs(dirname(self.global_hash_path), exist_ok=True)
        hash_value = hashlib.md5()
        label = ""

        if isinstance(self.input(), luigi.target.FileSystemTarget):
            input_file = Path(self.input().path)
            label = self.__helper_variant_label(input_file)

        elif isinstance(self.input(), (list, tuple, dict)):

            flattened_data = self.__get_flatten_data(self.input())
            var_label_name = []
            for item in flattened_data:
                input_file = Path(item.path)
                var_label_name.append(self.__helper_variant_label(input_file))
            label = "(" + (", ".join(var_label_name)) + ")" + " --> " + \
                self.__class__.__name__ if len(
                    ", ".join(var_label_name)) > 0 else self.__class__.__name__

        else:
            label = self.__class__.__name__

        label = "(" + label + "_" + name + ")"

        hash_value.update(label.encode())
        path = Path(self.global_hash_path + hash_value.hexdigest())
        if not path.is_file():
            with path.open(mode='w+') as hash_file:
                hash_file.write(label)
        return self.__class__.__name__ + "_" + "#" + hash_value.hexdigest() + "#" + "_" + name if label else self.__class__.__name__ + "_" + name

    def __helper_variant_label(self, input_file):
        input_filename = input_file.name
        try:
            _, lookup_hash, _ = input_filename.split("#", maxsplit=2)
            if len(lookup_hash) == 32:
                hash_file = Path(self.global_hash_path + lookup_hash)
                if hash_file.is_file():
                    with hash_file.open(mode='r') as f:
                        replacement_of_hash = f.read()
                        # label = "(" + replacement_of_hash + ")" + " --> " + self.__class__.__name__ if len(
                        #     input_filename) > 0 else self.__class__.__name__
                        label = replacement_of_hash
                        return label
                else:
                    raise ValueError
            else:
                raise ValueError

        except ValueError:
            label = input_filename
            return label


class CLSTask(luigi.Task, LuigiCombinator, CLSBaseTask):
    abstract = True


class CLSWrapperTask(luigi.WrapperTask, LuigiCombinator, CLSBaseTask):
    abstract = True


class CLSExternalTask(luigi.ExternalTask, LuigiCombinator, CLSBaseTask):
    abstract = True
