from ..template import OptimizeLots
import luigi
from examples.lot_sizing.implementations.lot_sizing_algorithms.silver_meal_heuristic import SilverMeal


class OptimizeLotsBySilverMeal(OptimizeLots):
    abstract = False

    def output(self):
        return luigi.LocalTarget('data/' + self._get_variant_label() + "-" + 'optimize_lots_by_silver_meal.txt')

    def run_optimizer(self, cost, demand):
        print("============= OptimizeLotsBySilverMeal: run")
        optimizer = SilverMeal()
        orders = optimizer.run(cost, demand)
        return orders
