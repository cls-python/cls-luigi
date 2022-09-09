import luigi
import inhabitation_task
from inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta, deep_str

from cls_python import FiniteCombinatoryLogic, Subtypes
from hello_world_10 import WriteFileTask
from repo_visualizer.dynamic_json_repo import DynamicJSONRepo
from repo_visualizer.static_json_repo import StaticJSONRepo


class SubstituteNameByAnneTask(luigi.Task, LuigiCombinator):
    abstract = False
    write_file_task = inhabitation_task.ClsParameter(tpe=WriteFileTask.return_type())

    def requires(self):
        return self.write_file_task()

    def output(self):
        return luigi.LocalTarget('pure_hello_anne.txt')

    def run(self):
        print("============= NameSubstitute: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text.replace('World', "Anne")
            outfile.write(text)


class SubstituteNameByJanTask(luigi.Task, LuigiCombinator):
    abstract = False

    write_file_task = inhabitation_task.ClsParameter(tpe=WriteFileTask.return_type())

    def requires(self):
        return self.write_file_task()

    def output(self):
        return luigi.LocalTarget('pure_hello_jan.txt')

    def run(self):
        print("============= NameSubstitute: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text.replace('World', "Jan")
            outfile.write(text)


class FinalTask(luigi.WrapperTask, LuigiCombinator):
    substitute_name = ClsParameter(tpe={1: SubstituteNameByJanTask.return_type(),
                                        "2": SubstituteNameByAnneTask.return_type()})
    config_domain = {1, "2"}

    def requires(self):
        return self.substitute_name()


if __name__ == '__main__':
    target = FinalTask.return_type()
    repository = RepoMeta.repository
    StaticJSONRepo(RepoMeta).dump_static_repo("../repo_visualizer")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if not actual is None or actual == 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]
    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_dict("../repo_visualizer")
        luigi.build(results, local_scheduler=False)  # für luigid: local_scheduler = True weglassen!
    else:
        print("No results!")
