import luigi
from cls.debug_util import deep_str
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter

import pandas as pd
import json

from lot_optimizers.groff_heuristic import GroffHeuristic
from lot_optimizers.wagner_within import WagnerWithin
from lot_optimizers.silver_meal_heuristic import SilverMeal
from lot_optimizers.least_unit_cost_method import LeastUnitCostMethod
from lot_optimizers.part_period_heuristic import PartPeriod


class GetCost(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/cost.json')

    def run(self):
        d = {
            "fixedCost" : 400,  # Bestellkosten
            "varCost" : 1,  # Lagerhaltungssatz
        }
        with open(self.output().path, 'w') as f:
            json.dump(d, f, indent=4)


class GetHistoricDemand(luigi.Task, LuigiCombinator):
    def output(self):
        print("GetHistoricDemand: output")
        return luigi.LocalTarget('data/historic_demand.csv')

    def run(self):
        print("====== GetHistoricDemand: run")
        with self.output().open('w') as f:
            f.write("1, 5, 7, 8, 9, 10, 14, 16, 19, 21, 19, 23, 24, 26, 26, "
                    "26, 28, 26, 28, 30")


class PredictDemand(luigi.Task, LuigiCombinator):
    abstract = True
    get_historic_demand = ClsParameter(tpe=GetHistoricDemand.return_type())
    prediction_horizon = 8

    def requires(self):
        return self.get_historic_demand()


class PredictDemandByAverage(PredictDemand):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/predicted_demand_by_average.pkl')

    def run(self):
        print("============= PredictDemandByAverage: run")
        with self.input().open() as infile:
            text = infile.read()
            l = [int(t) for t in text.split(",")]
            avg = int(sum(l) / len(l) + 0.5)
            predicted = [avg for i in range(self.prediction_horizon)]
            data = {'predicted_demand': predicted}
            df_predicted = pd.DataFrame(data)

            df_predicted.to_pickle(self.output().path)


class OptimizeLots(luigi.Task, LuigiCombinator):
    abstract = True
    predicted_demand = ClsParameter(tpe=PredictDemand.return_type())
    get_cost = ClsParameter(tpe=GetCost.return_type())

    def requires(self):
        return [self.get_cost(), self.predicted_demand()]

    def _get_cost(self):
        with open(self.input()[0].path, 'rb') as f:
            cost = json.load(f)
        return cost

    def _get_demand(self):
        demand_df = pd.read_pickle(self.input()[1].path)
        return list(demand_df['predicted_demand'])

    def run(self):
        cost = self._get_cost()
        demand = self._get_demand()

        orders = self.run_optimizer(cost, demand)

        with self.output().open('w') as f:
            f.write(str(list(orders)))

    def run_optimizer(self, cost, demand):
        return NotImplementedError()


class OptimizeLotsByGroff(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/optimize_lots_by_groff.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByGroff: run")
        optimizer = GroffHeuristic()
        orders = optimizer.run(cost, demand)
        return orders


class OptimizeLotsByWagnerWithin(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/optimize_lots_by_wagner_within.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByWagnerWithin: run")
        optimizer = WagnerWithin()
        orders = optimizer.run(cost, demand)
        return orders


class OptimizeLotsBySilverMeal(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/optimize_lots_by_silver_meal.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsBySilverMeal: run")
        optimizer = SilverMeal()
        orders = optimizer.run(cost, demand)
        return orders


class OptimizeLotsByLeastUnitCost(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/optimize_lots_by_least_unit_cost.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByLeastUnitCost: run")
        optimizer = LeastUnitCostMethod()
        orders = optimizer.run(cost, demand)
        return orders


class OptimizeLotsByPartPeriod(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/optimize_lots_by_part_period.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByPartPeriod: run")
        optimizer = PartPeriod()
        orders = optimizer.run(cost, demand)
        return orders


if __name__ == "__main__":
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = OptimizeLots.return_type()
    repository = RepoMeta.repository
    StaticJSONRepo(RepoMeta).dump_static_repo_json()
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    print(deep_str(inhabitation_result.rules))
    print("TREE:============")
    print(str())
    print("TREE:============")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if not actual is None or actual == 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]

    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=True, detailed_summary=True)
    else:
        print("No results!")
