import luigi
from inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta, deep_str
from cls_python import FiniteCombinatoryLogic, Subtypes


class Tabular(luigi.Task, LuigiCombinator):
    abstract = True


class TabularReadData(Tabular):
    abstract = False

    def output(self):
        return luigi.LocalTarget('TabularReadData.txt')

    def run(self):
        print("====== TabularReadData: run")
        with self.output().open('w') as f:
            f.write("id, value\\n1, 23")


class TabularTransformer1(Tabular):
    abstract = False
    tabular = ClsParameter(tpe=Tabular.return_type())

    def requires(self):
        return self.tabular()

    def output(self):
        return luigi.LocalTarget('TabularTransformer1.txt')

    def run(self):
        print("====== TabularTransformer1: run")
        with self.output().open('w') as f:
            f.write("id, value\\n1, 25")


class TabularTransformer2(Tabular):
    abstract = False
    tabular = ClsParameter(tpe=Tabular.return_type())

    def requires(self):
        return self.tabular()

    def output(self):
        return luigi.LocalTarget('TabularTransformer2.txt')

    def run(self):
        print("====== TabularTransformer2: run")
        with self.output().open('w') as f:
            f.write("id, value\\n1, 25")



class SomeClassifier(luigi.Task, LuigiCombinator):
    abstract = False
    tabular = ClsParameter(tpe=Tabular.return_type())

    def requires(self):
        return self.tabular()

    def output(self):
        return luigi.LocalTarget('SomeClassifier.txt')

    def run(self):
        print("====== SomeClassifier: run")
        with self.output().open('w') as f:
            f.write("SomeResult")


if __name__ == '__main__':
    from repo_visualizer.static_json_repo import StaticJSONRepo
    from repo_visualizer.dynamic_json_repo import DynamicJSONRepo


    target = SomeClassifier.return_type()
    print("Collecting Repo")
    rm = RepoMeta
    repository = rm.repository
    StaticJSONRepo(RepoMeta).dump_static_repo_json()


    subtpes = rm.subtypes
    print(deep_str(repository))

    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes),
                                 processes=1)
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
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results,
                    local_scheduler=False)  # für luigid: local_scheduler =
        # True weglassen!
    else:
        print("No results!")
