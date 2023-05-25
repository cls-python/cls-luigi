import luigi 
import sys 
from os.path import dirname
from os.path import join as pjoin
from os import makedirs
import hashlib
import flatdict
from collections.abc import Iterable
from pathlib import Path
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from inhabitation_task import ClsParameter, RepoMeta, LuigiCombinator

RESULT_PATH = "results/"

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


class StartNode(CLSTask):
    abstract = False
    
    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, ".dirs_created"))

    def run(self):
        makedirs(dirname(RESULT_PATH), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass


class SomeAbstractClass(CLSTask):
    abstract = True
    
    def requires(self):
        return {"start_node" : StartNode()}
    
    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "some_abstract_class_result"))
    
    def run(self):
        with open(self.output().path, 'w') as file:
            pass

class ConcreteClass4(SomeAbstractClass):
    abstract = False

class SomeAbstractAbstractClass(SomeAbstractClass):
    abstract = True
    
class ConcreteClass1(SomeAbstractAbstractClass):
    abstract = False

class ConcreteClass2(SomeAbstractAbstractClass):
    abstract = False
    
class ConcreteClass3(SomeAbstractAbstractClass):
    abstract = False
    
class SomeOtherAbstractAbstractClass(SomeAbstractClass):
    abstract = True
    
class ConcreteClassInAbstractChain(SomeOtherAbstractAbstractClass):
    abstract = False
    
class AbstractFromConcreteClassInChain(ConcreteClassInAbstractChain):
    abstract = True
    
class ConcreteClass5(AbstractFromConcreteClassInChain):
    abstract = False
    
class ConcreteClass6(AbstractFromConcreteClassInChain):
    abstract = False
    
class ConcreteClass7(AbstractFromConcreteClassInChain):
    abstract = False  

class UnrelatedAbstractClass(CLSTask):
    abstract = True
    some_class = ClsParameter(tpe=SomeAbstractClass.return_type())
    
    def requires(self):
        return {"some_class" : self.some_class()}
    
    def output(self):
       return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "unrelated_class_result"))
    
    def run(self):
         with open(self.output().path, 'w') as file:
            pass
    
class UnrelatedConcreteClass1(UnrelatedAbstractClass):
    abstract = False

class UnrelatedConcreteClass2(UnrelatedAbstractClass):
    abstract = False

class EndNode(CLSTask):
    abstract = False
    unreleated_class = ClsParameter(tpe=UnrelatedAbstractClass.return_type())
    
    
    def requires(self):
        return {"unrelated_class" : self.unreleated_class()}
    
    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "end_node_result"))
    
    
    def run(self):
        with open(self.output().path, 'w') as file:
            pass

class EndEndNode(luigi.Task, LuigiCombinator):
    pass

class WrapperTask(CLSWrapperTask):
    abstract = False
    config = ClsParameter(tpe={1: ConcreteClass1.return_type(),
                                        "2": ConcreteClass2.return_type()})
    config_domain = {1, "2"}
    def requires(self):
        return self.config()