import luigi
from inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from unique_task_pipeline_validator import UniqueTaskPipelineValidator


class Task1Abstract(luigi.Task, LuigiCombinator):
    abstract = True


class Task1A(Task1Abstract):
    abstract = False

    def run(self):
        print("Now Task1A")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task1A")

    def output(self):
        return luigi.LocalTarget("Task1A.txt")


class Task1B(Task1Abstract):
    abstract = False

    def run(self):
        print("Now Task1B")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task1B")

    def output(self):
        return luigi.LocalTarget("Task1B.txt")


class Task2(luigi.Task, LuigiCombinator):
    abstract = False
    task1 = ClsParameter(tpe=Task1Abstract.return_type())

    def requires(self):
        return self.task1()

    def run(self):
        print("Now Task2")

        with open(self.output().path, "w+") as outfile:
            outfile.write("Task2")

    def output(self):
        return luigi.LocalTarget("Task2.txt")


class Task3(luigi.Task, LuigiCombinator):
    abstract = False
    task1 = ClsParameter(tpe=Task1Abstract.return_type())
    task2 = ClsParameter(tpe=Task2.return_type())

    def requires(self):
        return [self.task1(), self.task2()]

    def run(self):
        print("Now Task3")

        with open(self.output().path, "w+") as outfile:
            outfile.write("Task3")

    def output(self):
        return luigi.LocalTarget("Task3.txt")


class Task4(luigi.Task, LuigiCombinator):
    abstract = False
    task3 = ClsParameter(tpe=Task3.return_type())
    task1 = ClsParameter(tpe=Task1Abstract.return_type())

    def requires(self):
        return [self.task1(), self.task3()]

    def run(self):
        print("Now Task4")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task4")

    def output(self):
        return luigi.LocalTarget("Task4.txt")



if __name__ == "__main__":

    target = Task4.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
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

    validator = UniqueTaskPipelineValidator([Task1Abstract])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    # results = [t() for t in inhabitation_result.evaluated[0:max_results]] # this is what we should NOT be using :)

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")




