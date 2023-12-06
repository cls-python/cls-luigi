import luigi
import os
from cls_luigi.inhabitation_task import LuigiCombinator, ClsParameter
import json
from pathlib import Path
import pandas as pd


class GetCost(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/cost.json')

    def run(self):
        d = {
            "fixedCost": 400,  # Bestellkosten
            "varCost": 1,  # Lagerhaltungssatz
        }
        os.makedirs('data', exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(d, f, indent=4)


class GetHistoricDemand(luigi.Task, LuigiCombinator):
    abstract = False

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

    def _get_variant_label(self):
        if isinstance(self.input()[1], luigi.LocalTarget):
            label = self.input()[1].path
            return Path(label).stem
