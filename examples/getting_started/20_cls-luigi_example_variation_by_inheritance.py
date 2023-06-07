import luigi
from string import Template

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

class WriteTemplateTask(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
            print("WriteTemplateTask: output")
            return luigi.LocalTarget('hello_world_template.st')

    def run(self):
        print("====== WriteTemplateTask: run")
        with self.output().open('w') as result:
            result.write("Hello World $name")

class SubstituteNameTask(luigi.Task, LuigiCombinator):
    abstract = True
    template_task = ClsParameter(tpe=WriteTemplateTask.return_type())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = ""

    def _set_name(self):
        # should be implemented in the concrete tasks
        pass

    def requires(self):
        return self.template_task()

    def output(self):
        return luigi.LocalTarget(self.__class__.__name__ + '_filled_template.txt')

    def run(self):
        self._set_name()
        with self.input().open() as input_template:
            template = Template(input_template.read())
            result = template.substitute(name=self.name)
            with self.output().open('w') as outfile:
                outfile.write(result)


class SubstituteNameByAnneTask(SubstituteNameTask):
    abstract = False

    def _set_name(self):
        self.name = "Anne"

class SubstituteNameByJanTask(SubstituteNameTask):
    abstract = False

    def _set_name(self):
        self.name = "Jan"


def main():
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = SubstituteNameTask.return_type()
    repository = RepoMeta.repository
    StaticJSONRepo(RepoMeta).dump_static_repo_json()
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
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
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
