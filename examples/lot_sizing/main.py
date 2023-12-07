# CLS-Lugi imports
from cls_luigi.inhabitation_task import RepoMeta
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls.debug_util import deep_str

from implementations.template import *

# Demand prediction algorithms
from implementations.predict_demand.average_demand import PredictDemandByAverage
from implementations.predict_demand.linear_regression import PredictDemandByLinearRegression


# Lot-sizing algorithms
from implementations.optimize_lots.silver_meal import OptimizeLotsBySilverMeal
from implementations.optimize_lots.least_unit_cost import OptimizeLotsByLeastUnitCost
from implementations.optimize_lots.part_period import OptimizeLotsByPartPeriod
from implementations.optimize_lots.wagner_whitin import OptimizeLotsByWagnerWhitin
from implementations.optimize_lots.groff import OptimizeLotsByGroff

if __name__ == "__main__":
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = OptimizeLots.return_type()
    repository = RepoMeta.repository
    StaticJSONRepo(RepoMeta).dump_static_repo_json()
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    print(deep_str(inhabitation_result.rules))
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
