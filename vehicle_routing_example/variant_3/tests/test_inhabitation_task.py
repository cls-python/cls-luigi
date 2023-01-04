import unittest
import luigi 
import sys 
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from inhabitation_task import *
from cls_tasks import * 


class AbstractMptopConfig(CLSTask):
    abstract = True

class SingleConcreteMptopConfig(AbstractMptopConfig):
    abstract = True

class AbstractConfigPack1(AbstractMptopConfig):
    abstract = True
    
class Pack1Config1(AbstractConfigPack1):
    abstract = False

class Pack1Config2(AbstractConfigPack1):
    abstract = False
    
class Pack1Config3(AbstractConfigPack1):
    abstract = False
    
class AbstractConfigPack2(AbstractMptopConfig):
    abstract = True
    
class ConcreteConfigInAbstractChain(AbstractConfigPack2):
    abstract = False
    
class AbstractConfigFromConcreteConfig(ConcreteConfigInAbstractChain):
    abstract = True
    
class Pack2Config1(AbstractConfigFromConcreteConfig):
    abstract = False
    
class Pack2Config2(AbstractConfigFromConcreteConfig):
    abstract = False
    
class Pack2Config3(AbstractConfigFromConcreteConfig):
    abstract = False  

class WrapperTask(CLSWrapperTask):
    abstract = False
    config = ClsParameter(tpe={1: Pack1Config1.return_type(),
                                        "2": Pack1Config2.return_type()})
    config_domain = {1, "2"}
    def requires(self):
        return self.config()

class TestRepositoryFilterMethods(unittest.TestCase):
    
    @classmethod
    def tearDownClass(cls):
         show_subtypes_dict()
    
    def test_get_list_of_all_upstream_classes_PackConfig3(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_classes(Pack1Config3),  [Pack1Config3, AbstractConfigPack1, AbstractMptopConfig, CLSTask])
        
    def test_get_list_of_all_upstream_classes_SingleConcreteMptopConfig(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_classes(SingleConcreteMptopConfig),  [SingleConcreteMptopConfig, AbstractMptopConfig, CLSTask])
    
    def test_get_list_of_all_upstream_classes_Pack2Config1(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_classes(Pack2Config1),  [Pack2Config1, AbstractConfigFromConcreteConfig, ConcreteConfigInAbstractChain, AbstractConfigPack2, AbstractMptopConfig, CLSTask])
        
    def test_get_list_of_all_upstream_abstract_classes_PackConfig3(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_abstract_classes(Pack1Config3), [AbstractConfigPack1, AbstractMptopConfig])
        
    def test_get_list_of_all_upstream_abstract_classes_Pack2Config1(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_abstract_classes(Pack2Config1), [AbstractConfigFromConcreteConfig, AbstractConfigPack2, AbstractMptopConfig])
        
    def test_get_list_of_all_upstream_abstract_classes_AbstractConfigFromConcreteConfig(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_abstract_classes(AbstractConfigFromConcreteConfig), [AbstractConfigFromConcreteConfig, AbstractConfigPack2, AbstractMptopConfig])
        
    def test_get_list_of_all_upstream_abstract_classes_ConcreteConfigInAbstractChain(self):
        self.assertListEqual(RepoMeta.get_list_of_all_upstream_abstract_classes(ConcreteConfigInAbstractChain), [AbstractConfigPack2, AbstractMptopConfig])
        

        
    
    
    
def show_subtypes_dict():
    print("SubTypes:")
    subtypes = RepoMeta.subtypes
    for item in subtypes:
        print("*****************")
        print("key: ", str(item), " :-> ", "value: ", str(subtypes[item]))
        print("*****************")
    
    
if __name__=="__main__":
    unittest.main()