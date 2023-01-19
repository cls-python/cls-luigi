import luigi
import luigi.configuration
import sys
import os
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from inhabitation_task import RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes

from importlib import import_module

def run_cls_luigi():
    # Create a configuration object
    config = luigi.configuration.get_config()
    # Set the log level
    config.set("logging", "level", "DEBUG")
        
    module = import_module("task_implementation")
    EndNode = getattr(module, "EndNode")    
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
        #no_schedule_error = luigi.build(results, local_scheduler=False, workers=3, detailed_summary=True)
        #return no_schedule_error
    else:
        print("No results!") 
        return False
    
    
if __name__=="__main__":
    run_cls_luigi()