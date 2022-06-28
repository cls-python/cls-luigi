import luigi
import inhabitation_task
from inhabitation_task import RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from hello_world import WriteFileTask
from pathlib import Path


class SubstituteNameTask(luigi.Task, inhabitation_task.LuigiCombinator):
    abstract = True
    write_file_task = inhabitation_task.ClsParameter(tpe=WriteFileTask.return_type())

    def requires(self):
        return self.write_file_task()


class SubstituteNameByAnneTask(SubstituteNameTask):
    abstract = False

    def output(self):
        return luigi.LocalTarget('pure_hello_anne.txt')

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
        return luigi.LocalTarget('pure_hello_jan.txt')

    def run(self):
        print("============= NameSubstituter: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text.replace('World', "Jan")
            outfile.write(text)


class AddHadiToSubstituteNameTask(luigi.Task, inhabitation_task.LuigiCombinator):
    abstract = False
    substituted_name = inhabitation_task.ClsParameter(tpe=SubstituteNameTask.return_type())

    def _get_variant_label(self):
        label = self.input().path
        return Path(label).stem

    def requires(self):
        return self.substituted_name()

    def output(self):
        return luigi.LocalTarget(self._get_variant_label() + "-hadi.txt")

    def run(self):
        print("============= NameSubstituter: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text + "\nand Hadi as well!"
            # text = text.replace('World', "Jan")
            outfile.write(text)





if __name__ == "__main__":
    target = AddHadiToSubstituteNameTask.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if not actual is None or actual == 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]
    if results:
        luigi.build(results, local_scheduler=True)  # für luigid: local_scheduler = True weglassen!
    else:
        print("No results!")
