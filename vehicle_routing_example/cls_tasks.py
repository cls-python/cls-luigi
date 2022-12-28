import hashlib
import flatdict
import sys
import luigi
from collections.abc import Iterable
from pathlib import Path
sys.path.append('../')
from inhabitation_task import LuigiCombinator



class CLSBaseTask():

    hash_map = {}

    def _get_variant_label(self):
        if isinstance(self.input(), luigi.LocalTarget):
            label = self.input().path

            return (Path(label).stem) + "-" + self.__class__.__name__ if len(Path(label).stem) > 0 else self.__class__.__name__

        elif isinstance(self.input(), dict):

            d = flatdict.FlatDict(self.input(), delimiter='.')
            flat_dict = dict(d)

            var_label_name = []
            for item in flat_dict.values():
                var_label_name.append(Path(item.path).stem)
            variant_label = ("-".join(var_label_name))
            if len(variant_label + "-" + self.__class__.__name__) <= 256:
                return variant_label + "-" + self.__class__.__name__ if len(variant_label) > 0 else self.__class__.__name__
            else:
                label_hash = str(
                    int(hashlib.sha1((variant_label).encode("utf-8")).hexdigest(), 16) % (10 ** 8))
                self.hash_map[label_hash] = var_label_name
                return label_hash + "-" + self.__class__.__name__ if len(label_hash) > 0 else self.__class__.__name__

        elif isinstance(self.input(), Iterable):
            var_label_name = list(map(
                lambda outputs: Path(outputs.path).stem, self.input()))
            return ("-".join(var_label_name)) + "-" + self.__class__.__name__ if len(("-".join(var_label_name))) > 0 else self.__class__.__name__


class CLSTask(luigi.Task, LuigiCombinator, CLSBaseTask):
    pass


class CLSWrapperTask(luigi.WrapperTask, LuigiCombinator, CLSBaseTask):
    pass


class CLSExternalTask(luigi.ExternalTask, LuigiCombinator, CLSBaseTask):
    pass
