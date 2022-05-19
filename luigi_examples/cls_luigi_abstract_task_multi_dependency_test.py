import luigi
from inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes


class Task1(luigi.Task, LuigiCombinator):
    abstract = False

    def run(self):
        print("Now Task1")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task1")

    def output(self):
        return luigi.LocalTarget("data/Task1.txt")


class Task2Abstract(luigi.Task, LuigiCombinator):
    abstract = True
    task1 = ClsParameter(tpe=Task1.return_type())

    def requires(self):
        return self.task1()


class Task2A(Task2Abstract):
    abstract = False

    def run(self):
        print("Now Task2A")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task2A")

    def output(self):
        return luigi.LocalTarget("data/Task2A.txt")


class Task2B(Task2Abstract):
    abstract = False

    def run(self):
        print("Now Task2B")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task2B")

    def output(self):
        return luigi.LocalTarget("data/Task2B.txt")


class Task3(luigi.Task, LuigiCombinator):
    abstract = False
    task2 = ClsParameter(tpe=Task2Abstract.return_type())

    def requires(self):
        return self.task2()

    def run(self):
        print("Now Task3")

        with open(self.output().path, "w+") as outfile:
            outfile.write("Task3")

    def output(self):
        return luigi.LocalTarget("data/Task3.txt")


class Task4(luigi.Task, LuigiCombinator):
    abstract = False
    task2 = ClsParameter(tpe=Task2Abstract.return_type())
    # task3 = ClsParameter(tpe=Task3.return_type())

    def requires(self):
        return [self.task2()]

    def run(self):
        print("Now Task4")

        with open(self.output().path, "w+") as outfile:
            outfile.write("Task4")

    def output(self):
        return luigi.LocalTarget("data/Task4.txt")


class Task5(luigi.Task, LuigiCombinator):
    abstract = False
    task4 = ClsParameter(tpe=Task4.return_type())

    def requires(self):
        return self.task4()

    def run(self):
        print("Now Task5")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task5")

    def output(self):
        return luigi.LocalTarget("data/Task5.txt")


class Task6(luigi.Task, LuigiCombinator):
    abstract = False
    task5 = ClsParameter(tpe=Task5.return_type())
    task2 = ClsParameter(tpe= Task2Abstract.return_type())

    def requires(self):
        return [self.task5(), self.task2()]

    def run(self):
        print("Now Task6")
        with open(self.output().path, "w+") as outfile:
            outfile.write("Task6")

    def output(self):
        return luigi.LocalTarget("data/Task6.txt")


class EndNode(luigi.WrapperTask, LuigiCombinator):
    task6 = ClsParameter(tpe=Task6.return_type())

    def requires(self):
        return self.task6()


if __name__ == '__main__':
    target = EndNode.return_type()
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
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]
    if results:
        print("Number of results", max_results)
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
