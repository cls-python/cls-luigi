import luigi

from inhabitation_task import RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes


from unique_task_pipeline_validator import UniqueTaskPipelineValidator

from _70_ML_example_variation_point_multi_usage import EvaluateRegressionModel, FitTransformScaler, TrainRegressionModel

if __name__ == '__main__':

    from repo_visualizer.static_json_repo import StaticJSONRepo

    target = EvaluateRegressionModel.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    StaticJSONRepo(RepoMeta).dump_static_repo_json()
    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    from repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    validator = UniqueTaskPipelineValidator([TrainRegressionModel, FitTransformScaler])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
