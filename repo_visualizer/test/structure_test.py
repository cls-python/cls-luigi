import sys
from os.path import dirname
from os.path import join as pjoin
from os import makedirs

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from inhabitation_task import *
from cls_base_tasks import *
from unique_task_pipeline_validator import UniqueTaskPipelineValidator
from inhabitation_task import ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from repo_visualizer.static_json_repo import StaticJSONRepo
from repo_visualizer.dynamic_json_repo import DynamicJSONRepo

RESULT_PATH = "results/"


class StartNode(CLSTask):
    abstract = False

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, "start_node"))

    def run(self):
        makedirs(dirname(RESULT_PATH), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass


class TaskA(CLSTask):
    abstract = False
    start = ClsParameter(tpe=StartNode.return_type())

    def requires(self):
        return {"start_node": self.start()}

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, "TaskA"))

    def run(self):
        makedirs(dirname(RESULT_PATH), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass


class AbstractA(CLSTask):
    abstract = True

    taskA = ClsParameter(tpe=TaskA.return_type())

    def requires(self):
        return {"start_node": self.taskA()}

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "AbstractA"))

    def run(self):
        with open(self.output().path, 'w') as file:
            pass


class ConcreateA_AbstractA(AbstractA):
    abstract = False


class ConcreateB_AbstractA(AbstractA):
    abstract = False


class ConcreateC_AbstractA(AbstractA):
    abstract = False


class TaskB(CLSTask):
    abstract = False
    absA = ClsParameter(tpe=AbstractA.return_type())

    def requires(self):
        return {"start_node": self.absA()}

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, "TaskB"))

    def run(self):
        makedirs(dirname(RESULT_PATH), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass


class TaskC(CLSTask):
    abstract = False
    absA = ClsParameter(tpe=AbstractA.return_type())

    def requires(self):
        return {"start_node": self.absA()}

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, "TaskC"))

    def run(self):
        makedirs(dirname(RESULT_PATH), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass

class Config(CLSWrapperTask):
    abstract = False
    config = ClsParameter(tpe={1: ConcreateA_AbstractA.return_type(),
                                        2: ConcreateB_AbstractA.return_type(),
                               3: ConcreateC_AbstractA.return_type(),
                               4: TaskC.return_type()})
    config_domain = {1, 2, 3, 4}
    def requires(self):
        return self.config()

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "WrapperTask"))

    def run(self):
        with open(self.output().path, 'w') as file:
            pass


class AbstractB(CLSTask):
    abstract = True

    taskA = ClsParameter(tpe=Config.return_type())

    def requires(self):
        return {"start_node": self.taskA()}

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "AbstractB"))

    def run(self):
        with open(self.output().path, 'w') as file:
            pass


class ConcreateA_AbstractB(AbstractB):
    abstract = False


class ConcreateB_AbstractB(AbstractB):
    abstract = False


class ConcreateC_AbstractB(AbstractB):
    abstract = False



class EndNode(CLSTask):
    abstract = False
    config = ClsParameter(tpe=Config.return_type())
    taskb = ClsParameter(tpe=TaskB.return_type())
    abstractb = ClsParameter(tpe=AbstractB.return_type())

    def requires(self):
        return {"config": self.config(),
                "taskb": self.taskb(),
                "abstractb": self.abstractb()}

    def output(self):
        return luigi.LocalTarget(pjoin(RESULT_PATH, self._get_variant_label() + "-" "end_node_result"))

    def run(self):
        with open(self.output().path, 'w') as file:
            pass


if __name__ == "__main__":
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
    validator = UniqueTaskPipelineValidator([AbstractA])
    results = [t() for t in inhabitation_result.evaluated[0:max_results]
               if validator.validate(t())]
    if results:

        DynamicJSONRepo(results).dump_dynamic_pipeline_json()

        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        no_schedule_error = luigi.build(results, local_scheduler=True, detailed_summary=True)
    else:
        print("No results!")
