import sys
sys.path.append('../../')

import luigi
from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls_luigi.cls_tasks import ClsTask
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls.debug_util import deep_str
from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo


class WriteFileTask(ClsTask):
    abstract = False

    def output(self):
        print("WriteFileTask: output")
        return self.create_result_file('pure_hello_world.txt')

    def run(self):
        print("====== WriteFileTask: run")
        with self.output().open('w') as f:
            f.write("Hello World")

class SubstituteNameTask(ClsTask):
    abstract = True
    write_file_task = ClsParameter(tpe=WriteFileTask.return_type())

    def requires(self):
        return self.write_file_task()


class SubstituteNameByAnneTask(SubstituteNameTask):
    abstract = False

    def output(self):
        return self.create_result_file('pure_hello_anne.txt')

    def run(self):
        print("============= NameSubstituter: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text.replace('World', "Anne")
            outfile.write(text)


class SubstituteNameByJanTask(SubstituteNameTask):
    abstract = False

    def output(self):
        return self.create_result_file('pure_hello_jan.txt')

    def run(self):
        print("============= NameSubstituter: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text.replace('World', "Jan")
            outfile.write(text)


def main():

    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = SubstituteNameTask.return_type()
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
