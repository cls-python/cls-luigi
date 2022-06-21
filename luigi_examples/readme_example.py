import luigi
import inhabitation_task


from inhabitation_task import RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes, deep_str


class WriteFileTask(luigi.Task, inhabitation_task.LuigiCombinator):

    def output(self):
        print("WriteFileTask: output")
        return luigi.LocalTarget('pure_hello_world.txt')

    def run(self):
        print("====== WriteFileTask: run")
        with self.output().open('w') as f:
            f.write("Hello World")


class SubstituteNameTask(luigi.Task, inhabitation_task.LuigiCombinator):
    abstract = True

    def requires(self):
        return self.write_file_task()


class SubstituteNameByAnneTask(SubstituteNameTask):
    abstract = False
    write_file_task = inhabitation_task.ClsParameter(tpe=WriteFileTask.return_type())

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
    write_file_task = inhabitation_task.ClsParameter(tpe=WriteFileTask.return_type())

    def output(self):
        return luigi.LocalTarget('pure_hello_jan.txt')

    def run(self):
        print("============= NameSubstituter: run")
        with self.input().open() as infile:
            text = infile.read()

        with self.output().open('w') as outfile:
            text = text.replace('World', "Jan")
            outfile.write(text)


class FinalNode(luigi.WrapperTask, inhabitation_task.LuigiCombinator):
    substitute_name = inhabitation_task.ClsParameter(tpe=SubstituteNameTask.return_type())

    def requires(self):
        return self.substitute_name()


if __name__ == "__main__":
    target = FinalNode.return_type()
    print("Collecting Repo")
    rm = RepoMeta
    repository = rm.repository
    subtpes = rm.subtypes
    print(deep_str(repository))

    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]
    if results:
        print("Number of results", max_results)
        print("Run Pipelines")
        luigi.build(results, local_scheduler=True)  # für luigid: local_scheduler = True weglassen!
    else:
        print("No results!")
