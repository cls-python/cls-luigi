import json
import luigi

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes


class EstimateYInterceptAndSlope(luigi.Task, LuigiCombinator):
    abstract = False

    hyper_parameters = {
        "slope": {"range": [-1, 1], "default": 1},
        "y-intercept": {"range": [-1, 1], "default": 1}
    }

    def output(self):
        return luigi.LocalTarget('EstimateYInterceptAndSlope.json')

    def run(self):
        print("====== EstimateYInterceptAndSlope: run")
        p = {}
        for k in self.hyper_parameters.keys():
            p[k] = self.hyper_parameters[k]["default"]
        with self.output().open('w') as result:
            json.dump(p, result)


class Shifter(luigi.Task, LuigiCombinator):
    abstract = False
    estimate_y_intercept_and_slope = \
        ClsParameter(tpe=EstimateYInterceptAndSlope.return_type())

    hyper_parameters = {
        "shifter": {
            "values": ["shift0", "shift10", "shift20"],
            "default": "shift10"}
            }

    def requires(self):
        return [self.estimate_y_intercept_and_slope()]

    def output(self):
        return luigi.LocalTarget("Shifter.json")

    def run(self):
        print("====== Shifter: run")
        with open(self.input()[0].path, 'rb') as f:
            input_data = json.load(f)

            shifter = self.hyper_parameters["shifter"]["default"]
            if shifter == "shift0":
                s = 0
            elif shifter == "shift10":
                s = 10
            elif shifter == "shift20":
                s = 20

            y_intercept = input_data['y-intercept'] + s
            slope = input_data['slope']

        with self.output().open('w') as result:
            json.dump({"y-intercept": y_intercept, "slope": slope}, result)


class EstimateXValue(luigi.Task, LuigiCombinator):
    abstract = False
    shifter = ClsParameter(tpe=Shifter.return_type())

    hyper_parameters = {
        "x-value": {"range": [0, 10], "default": 5}
    }

    def requires(self):
        return [self.shifter()]

    def output(self):
        return luigi.LocalTarget('EstimateXValue.json')

    def run(self):
        print("====== EstimateYinterceptAndSlope: run")
        with open(self.input()[0].path, 'rb') as f:
            input_data = json.load(f)

            x_value = self.hyper_parameters['x-value']['default']
            slope = input_data['slope']
            y_intercept = input_data['y-intercept']

            v = y_intercept + slope * x_value

        with self.output().open('w') as result:
            json.dump({"x_value": x_value, "result" : v}, result)


if __name__ == "__main__":
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = EstimateXValue.return_type()
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
        no_schedule_error = luigi.build(results, local_scheduler=True,
                                        workers=2, detailed_summary=True)
    else:
        print("No results!")




