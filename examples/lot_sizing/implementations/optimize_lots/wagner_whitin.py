from ..template import OptimizeLots
import luigi
from examples.lot_sizing.implementations.lot_sizing_algorithms.wagner_whitin_algorithm import WagnerWhitin




class OptimizeLotsByWagnerWhitin(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_wagner_within.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsByWagnerWithin: run")
        optimizer = WagnerWhitin()
        orders = optimizer.run(cost, demand)
        return orders
