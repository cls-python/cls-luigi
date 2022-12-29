import luigi
from tour_planning_tasks import *
import os
from os.path import join as pjoin


sys.path.append('../')
sys.path.append('../../')
from unique_task_pipeline_validator import UniqueTaskPipelineValidator
from inhabitation_task import ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from repo_visualizer.static_json_repo import StaticJSONRepo
from repo_visualizer.dynamic_json_repo import DynamicJSONRepo

class RunCLSPipelines(globalConfig, luigi.Task):

    def output(self):
        return {"cls_pipelines_done" : luigi.LocalTarget(pjoin(str(self.result_path), ".cls_pipelines_done"))}

    def run(self):
        target = CreateHashMapResult.return_type()
        repository = RepoMeta.repository
            
        fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
        inhabitation_result = fcl.inhabit(target)
        max_tasks_when_infinite = 10
        actual = inhabitation_result.size()
        max_results = max_tasks_when_infinite
        if not actual is None or actual == 0:
            max_results = actual
        validator = UniqueTaskPipelineValidator(
            [AbstractGatherAndIntegratePhase, AbstractMptopConfig, AbstractScoringPhase, AbstractRoutingPhase, AbstractSolverPhase])
        results = [t() for t in inhabitation_result.evaluated[0:max_results]
                if validator.validate(t())]
        if results:
            print("Number of results", max_results)
            print("Number of results after filtering", len(results))
            print("Run Pipelines")
            no_schedule_error = luigi.build(results, local_scheduler=False, detailed_summary=True)
            if no_schedule_error:
                with open(self.output()["cls_pipelines_done"].path, "w") as file:
                    file.write(" ")
        else:
            print("No results!")


class FilterBestResult(globalConfig, luigi.Task):
    best_result = ("NONE", 0)
    
    def requires(self):
        return RunCLSPipelines()

    def output(self):
        return {"best_result": luigi.LocalTarget(pjoin(str(self.best_result_path), "best_result.txt"))}

    def run(self):
        
        for filename in os.listdir(str(self.solver_result_path)):
            if filename.endswith("solver_result.txt"):
                with open(os.path.join(str(self.solver_result_path), filename), "r") as solver_result:
                    objVal = 0
                    for line in solver_result:
                        if line.startswith("\"ObjVal\""):
                            objVal = int(line.split(":")[1].replace(",", ""))
                    if objVal < self.best_result[1]:
                        self.best_result = (filename, objVal)
        with open(self.output()["best_result"].path, "w") as result:
            result.write(str(self.best_result))
        
        


if __name__ == '__main__':
    luigi.build([FilterBestResult()],local_scheduler=False, detailed_summary=True)