from ..template import OptimizeLots
import luigi
from examples.lot_sizing.implementations.lot_sizing_algorithms.least_unit_cost_method import LeastUnitCostMethod


class OptimizeLotsByLeastUnitCost(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_least_unit_cost.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByLeastUnitCost: run")
        optimizer = LeastUnitCostMethod()
        orders = optimizer.run(cost, demand)
        return orders
