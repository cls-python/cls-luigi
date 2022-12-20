import sys
import luigi
from collections.abc import Iterable
from pathlib import Path
from os.path import join as pjoin
sys.path.append('../')
from inhabitation_task import ClsParameter, LuigiCombinator

class CLSBaseTask():

    def _get_variant_label(self):
        if isinstance(self.input(), luigi.LocalTarget):
            label = self.input().path
            return (Path(label).stem) + "-" + self.__class__.__name__

        elif isinstance(self.input(), dict):
            print("###################")
            print(self.input())
            print("++++++++++++++++++")
            var_label_name = []
            for item in self.input().values():
                var_label_name.append(Path(item.path).stem)
            print("get_variant_done !!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return ("-".join(var_label_name)) + "-" + self.__class__.__name__

        elif isinstance(self.input(), Iterable):
            var_label_name = list(map(
                lambda outputs: Path(outputs.path).stem, self.input()))
            return ("-".join(var_label_name)) + "-" + self.__class__.__name__

class CLSTask(luigi.Task, LuigiCombinator, CLSBaseTask):
    pass


class CLSWrapperTask(luigi.WrapperTask, LuigiCombinator, CLSBaseTask):
    pass


class CLSExternalTask(luigi.ExternalTask, LuigiCombinator, CLSBaseTask):
    pass


