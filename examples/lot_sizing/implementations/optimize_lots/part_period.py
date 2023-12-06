from ..template import OptimizeLots
import luigi
from examples.lot_sizing.implementations.lot_sizing_algorithms.part_period_heuristic import PartPeriod


class OptimizeLotsByPartPeriod(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_part_period.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByPartPeriod: run")
        optimizer = PartPeriod()
        orders = optimizer.run(cost, demand)
        return orders
