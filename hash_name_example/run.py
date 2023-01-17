import luigi 
import sys 
from os.path import dirname
from os.path import join as pjoin
from os import makedirs
from cls_tasks import * 
sys.path.append('../')
from inhabitation_task import *
from unique_task_pipeline_validator import UniqueTaskPipelineValidator
from inhabitation_task import ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
RESULT_PATH = "results/"

class StartNode(CLSTask):
    abstract = False
    
    def output(self):
        return [luigi.LocalTarget(pjoin(RESULT_PATH,  self._get_variant_filename(".dirs_created"))), luigi.LocalTarget(pjoin(RESULT_PATH,  self._get_variant_filename(".dirs_created_2")))]

    def run(self):
        makedirs(dirname(RESULT_PATH), exist_ok=True)
        with open(self.output()[0].path, 'w') as file:
            pass
        with open(self.output()[1].path, 'w') as file:
            pass


class SomeAbstractClass(CLSTask):
    abstract = True
    
    def requires(self):
        return {"start_node" : StartNode()}
    
    def output(self):
        return {"SomeAbstractClass_result" : luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_filename("some_abstract_class_result"))), "start_node_result" : self.input()}
    
    def run(self):
        with open(self.output()["SomeAbstractClass_result"].path, 'w') as file:
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
        return [self.some_class(), StartNode()]
    
    def output(self):
       return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_filename("unrelated_class_result")))
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
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_filename("end_node_result")))
    
    
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
    
    
def main():
    target = EndNode.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if not actual is None or actual == 0:
        max_results = actual
    validator = RepoMeta.get_unique_abstract_task_validator()
    results = [t() for t in inhabitation_result.evaluated[0:max_results]
               if validator.validate(t())]
    if results:        
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        no_schedule_error = luigi.build(results, local_scheduler=True, workers=2, detailed_summary=True)
        return no_schedule_error
    else:
        print("No results!") 
        return False
    

if __name__=="__main__":
    main()