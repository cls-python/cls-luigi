from ..template import OptimizeLots
import luigi
from examples.lot_sizing.implementations.lot_sizing_algorithms.groff_heuristic import GroffHeuristic


class OptimizeLotsByGroff(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_groff.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByGroff: run")
        optimizer = GroffHeuristic()
        orders = optimizer.run(cost, demand)
        return orders
