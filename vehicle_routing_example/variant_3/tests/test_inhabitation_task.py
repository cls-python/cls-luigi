import unittest
import luigi 
import sys 
from os.path import dirname, basename, abspath
from os.path import join as pjoin
from os import makedirs
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from inhabitation_task import *
from cls_tasks import * 
from unique_task_pipeline_validator import UniqueTaskPipelineValidator
from inhabitation_task import ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from repo_visualizer.static_json_repo import StaticJSONRepo
from repo_visualizer.dynamic_json_repo import DynamicJSONRepo

RESULT_PATH = "results/"

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

class TestRepositoryFilterMethods(unittest.TestCase):
    
    @classmethod
    def tearDownClass(cls):
         show_repository_and_subtypes_dict()
    
    def test_get_list_of_all_upstream_classes_ConcreteClass3(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_classes(ConcreteClass3),  [ConcreteClass3, SomeAbstractAbstractClass, SomeAbstractClass, CLSTask])
        
    def test_get_list_of_all_upstream_classes_ConcreteClass4(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_classes(ConcreteClass4),  [ConcreteClass4, SomeAbstractClass, CLSTask])
    
    def test_get_list_of_all_upstream_classes_ConcreteClass5(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_classes(ConcreteClass5),  [ConcreteClass5, AbstractFromConcreteClassInChain, ConcreteClassInAbstractChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, CLSTask])
        
    def test_get_list_of_all_upstream_abstract_classes_ConcreteClass3(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(ConcreteClass3), [SomeAbstractAbstractClass, SomeAbstractClass, CLSTask])
        
    def test_get_list_of_all_upstream_abstract_classes_ConcreteClass5(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(ConcreteClass5), [AbstractFromConcreteClassInChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, CLSTask])
        
    def test_get_list_of_all_upstream_abstract_classes_AbstractFromConcreteClassInChain(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(AbstractFromConcreteClassInChain), [AbstractFromConcreteClassInChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, CLSTask])
        
    def test_get_list_of_all_upstream_abstract_classes_ConcreteClassInAbstractChain(self):
        self.assertListEqual(RepoMeta._get_list_of_all_upstream_abstract_classes(ConcreteClassInAbstractChain), [SomeOtherAbstractAbstractClass, SomeAbstractClass, CLSTask])
        
    def test_get_all_upstream_classes_ConcreteClass1(self):
        self.assertTupleEqual(RepoMeta._get_all_upstream_classes(ConcreteClass1), (ConcreteClass1, [SomeAbstractAbstractClass, SomeAbstractClass, CLSTask]))
        
    def test_get_set_of_all_downstream_classes_ConcreteClassInAbstractChain(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_classes(ConcreteClassInAbstractChain), (ConcreteClassInAbstractChain, {AbstractFromConcreteClassInChain, ConcreteClass5, ConcreteClass6, ConcreteClass7}))
    
    def test_get_set_of_all_downstream_classes_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_classes(SomeAbstractClass), (SomeAbstractClass, {SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, ConcreteClass1, ConcreteClass2, ConcreteClass3, ConcreteClassInAbstractChain, AbstractFromConcreteClassInChain, ConcreteClass5, ConcreteClass6, ConcreteClass7, ConcreteClass4}))
        
    def test_get_all_downstream_abstract_classes_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_abstract_classes(SomeAbstractClass), (SomeAbstractClass, {SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, AbstractFromConcreteClassInChain}))
        
    def test_get_all_downstream_abstract_classes_UnrelatedAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_abstract_classes(UnrelatedAbstractClass), (UnrelatedAbstractClass, set()))

    def test_get_class_chain_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_class_chain(SomeAbstractClass), (SomeAbstractClass, [CLSTask], {SomeAbstractAbstractClass, ConcreteClass1, ConcreteClass2, ConcreteClass3, SomeOtherAbstractAbstractClass, ConcreteClassInAbstractChain, AbstractFromConcreteClassInChain, ConcreteClass5, ConcreteClass6, ConcreteClass7, ConcreteClass4}))
        
    def test_get_class_chain_ConcreteClass5(self):
        self.assertTupleEqual(RepoMeta._get_abstract_class_chain(ConcreteClass5), (ConcreteClass5, [AbstractFromConcreteClassInChain, ConcreteClassInAbstractChain, SomeOtherAbstractAbstractClass, SomeAbstractClass, CLSTask], set()))
        
    def test_get_abstract_class_chain_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_abstract_class_chain(SomeAbstractClass), (SomeAbstractClass, [CLSTask], {SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, AbstractFromConcreteClassInChain}))
    
    def test_get_maximal_shared_upper_classes_ConcreteClass5_ConcreteClass6(self):
        self.assertListEqual(RepoMeta._get_maximal_shared_upper_classes([ConcreteClass5, ConcreteClass6]), [CLSTask, SomeAbstractClass, SomeOtherAbstractAbstractClass, ConcreteClassInAbstractChain, AbstractFromConcreteClassInChain])
        
    def test_get_maximal_shared_upper_classes_ConcreteClass2_ConcreteClass7(self):
        self.assertListEqual(RepoMeta._get_maximal_shared_upper_classes([ConcreteClass2, ConcreteClass7]), [CLSTask, SomeAbstractClass])
      
    def test_get_all_downstream_concrete_classes_ConcreteClassInAbstractChain(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_concrete_classes(ConcreteClassInAbstractChain), (ConcreteClassInAbstractChain, {ConcreteClass5, ConcreteClass6, ConcreteClass7}))
        
    def test_get_all_downstream_concrete_classes_SomeAbstractClass(self):
        self.assertTupleEqual(RepoMeta._get_all_downstream_concrete_classes(SomeAbstractClass), (SomeAbstractClass, {ConcreteClass1, ConcreteClass2, ConcreteClass3, ConcreteClass4 ,ConcreteClass5, ConcreteClass6, ConcreteClass7, ConcreteClassInAbstractChain}))
      
    def test_delete_related_combinators_1(self):
        repository = RepoMeta.repository
        to_remove = []
        reference = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):
                    
                if issubclass(item.cls, (UnrelatedConcreteClass2, ConcreteClass4,ConcreteClass1, ConcreteClass2, ConcreteClass3)):
                    to_remove.append(item)
                
        for item in to_remove:
            reference.pop(item)
            
        self.assertDictEqual(reference, RepoMeta._delete_related_combinators([UnrelatedConcreteClass2, ConcreteClass4,ConcreteClass1, ConcreteClass2, ConcreteClass3]))
    
    def test_delete_related_combinators_2(self):
        repository = RepoMeta.repository
        to_remove = []
        reference = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):
                    
                if issubclass(item.cls, (ConcreteClass1, UnrelatedConcreteClass1)):
                    to_remove.append(item)
                
        for item in to_remove:
            reference.pop(item)
            
        self.assertDictEqual(reference, RepoMeta._delete_related_combinators([ConcreteClass1, UnrelatedConcreteClass1]))
        
        
    def test_filtered_repository_SomeAbstractAbstractClass(self):
        new_repo = RepoMeta._delete_related_combinators([ConcreteClass4, ConcreteClassInAbstractChain, ConcreteClass5, ConcreteClass6, ConcreteClass7])
            
        self.assertDictEqual(RepoMeta.filtered_repository([SomeAbstractAbstractClass]), new_repo)
        
    def test_filtered_repository_SomeOtherAbstractAbstractClass_UnrelatedConcreteClass1(self):
        repository = RepoMeta.repository
        to_remove = []
        new_repo = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):
                    
                if issubclass(item.cls, (UnrelatedConcreteClass2, ConcreteClass4,ConcreteClass1, ConcreteClass2, ConcreteClass3)):
                    to_remove.append(item)
                
        for item in to_remove:
            new_repo.pop(item)
        
        self.assertDictEqual(RepoMeta.filtered_repository([SomeOtherAbstractAbstractClass, UnrelatedConcreteClass1]), new_repo)     
        
    
    def test_filtered_repository_with_deep_filter(self):
        repository = RepoMeta.repository
        to_remove = []
        new_repo = repository.copy()
        for item in repository:
            if not isinstance(item, RepoMeta.ClassIndex):
                    
                if issubclass(item.cls, (ConcreteClass1, UnrelatedConcreteClass1, ConcreteClass5, ConcreteClass6, ConcreteClass7)):
                    to_remove.append(item)
                
        for item in to_remove:
            new_repo.pop(item)
            
        self.assertDictEqual(RepoMeta.filtered_repository([(SomeAbstractClass, [SomeAbstractAbstractClass, ConcreteClassInAbstractChain, ConcreteClass4]), (SomeAbstractAbstractClass, [ConcreteClass2, ConcreteClass3]), UnrelatedConcreteClass2]), new_repo)
        
    def test_run_cls_luigi(self):
        result = run_cls_luigi()
        self.assertTrue(result, "luigi returned a scheduling error")
          
            
    
def show_repository_and_subtypes_dict():
    
    print("Repo: ")
    repository = RepoMeta.repository
    for item in repository:
        print("#################")
        print("key: ", str(item), " :-> ", "value: ", str(repository[item]))
        print("#################")
        
    print("SubTypes:")
    subtypes = RepoMeta.subtypes
    for item in subtypes:
        print("*****************")
        print("key: ", str(item), " :-> ", "value: ", str(subtypes[item]))
        print("*****************")

def run_cls_luigi():
    target = EndNode.return_type()
    repository = RepoMeta.repository

    StaticJSONRepo(RepoMeta).dump_static_repo_json()

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if not actual is None or actual == 0:
        max_results = actual
    validator = UniqueTaskPipelineValidator(
        [SomeAbstractClass, SomeAbstractAbstractClass, SomeOtherAbstractAbstractClass, AbstractFromConcreteClassInChain, UnrelatedAbstractClass])
    results = [t() for t in inhabitation_result.evaluated[0:max_results]
               if validator.validate(t())]
    if results:
        
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        no_schedule_error = luigi.build(results, local_scheduler=True, detailed_summary=True)
        return no_schedule_error
    else:
        print("No results!") 
        return False
    
    
if __name__=="__main__":
    unittest.main()