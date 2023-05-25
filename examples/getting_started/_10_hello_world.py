import luigi
import inhabitation_task
from inhabitation_task import RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes


class WriteFileTask(luigi.Task, inhabitation_task.LuigiCombinator):

    def output(self):
        print("WriteFileTask: output")
        return luigi.LocalTarget('pure_hello_world.txt')

    def run(self):
        print("====== WriteFileTask: run")
        with self.output().open('w') as f:
            f.write("Hello World")


if __name__ == "__main__":
    target = WriteFileTask.return_type()
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
