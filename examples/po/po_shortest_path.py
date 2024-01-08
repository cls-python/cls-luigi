import luigi
import numpy as np
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from base_task import BaseTaskClass
from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
import pyepo

from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from global_parameters import GlobalParameters


class GenerateAndSplitDataSet(BaseTaskClass):
    abstract = False

    def run(self):
        features, costs = pyepo.data.shortestpath.genData(
            num_data=5000,
            num_features=5,
            grid=(5, 5),
            deg=6,
            noise_width=0.5,
            seed=self.global_params.seed
        )

        x_train, x_test, c_train, c_test = train_test_split(
            features,
            costs,
            test_size=1000,
            random_state=self.global_params.seed
        )

        self.dump_pickle(x_train, self.output()["x_train"].path)
        self.dump_pickle(x_test, self.output()["x_test"].path)
        self.dump_pickle(c_train, self.output()["c_train"].path)
        self.dump_pickle(c_test, self.output()["c_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "c_train": self.get_luigi_local_target_with_task_id("c_train.pkl"),
            "c_test": self.get_luigi_local_target_with_task_id("c_test.pkl")
        }


class MultiOutputRegressionModel(BaseTaskClass):
    abstract = True
    split_dataset = ClsParameter(tpe=GenerateAndSplitDataSet.return_type())

    regressor = None

    def requires(self):
        return self.split_dataset()

    def output(self):
        return {
            "predictions": self.get_luigi_local_target_with_task_id("predictions.pkl"),
            "model": self.get_luigi_local_target_with_task_id("model.pkl")
        }

    def _load_split_input_dataset(self):
        x_train = self.load_pickle(self.input()["x_train"].path)
        x_test = self.load_pickle(self.input()["x_test"].path)
        c_train = self.load_pickle(self.input()["c_train"].path)
        c_test = self.load_pickle(self.input()["c_test"].path)
        return x_train, x_test, c_train, c_test

    def run(self):
        x_train, x_test, c_train, c_test = self._load_split_input_dataset()

        self._init_regressor()
        self.regressor.fit(x_train, c_train)

        predictions = self.regressor.predict(x_test)
        self.dump_pickle(predictions, self.output()["predictions"].path)
        self.dump_pickle(self.regressor, self.output()["model"].path)

    def _init_regressor(self):
        return NotImplementedError


class LinearRegressionModel(MultiOutputRegressionModel):
    abstract = False

    def _init_regressor(self):
        self.regressor = MultiOutputRegressor(
            estimator=LinearRegression(
                n_jobs=self.global_params.n_jobs
            ),
            n_jobs=self.global_params.n_jobs,
        )


class RandomForestModel(MultiOutputRegressionModel):
    abstract = False

    def _init_regressor(self):
        self.regressor = MultiOutputRegressor(
            estimator=RandomForestRegressor(
                n_jobs=self.global_params.n_jobs,
                random_state=self.global_params.seed
            ),
            n_jobs=self.global_params.n_jobs,
        )


class OptimizationModel(BaseTaskClass):
    abstract = True
    split_dataset = ClsParameter(tpe=GenerateAndSplitDataSet.return_type())
    predictions = ClsParameter(tpe=MultiOutputRegressionModel.return_type())

    optimizer = None

    def requires(self):
        return {
            "split_dataset": self.split_dataset(),
            "predictions": self.predictions()
        }

    def _init_optimizer(self):
        return NotImplementedError

    def output(self):
        return {
            "true_solutions": self.get_luigi_local_target_with_task_id("true_solutions.pkl"),
            "true_objective_value": self.get_luigi_local_target_with_task_id("true_objective_value.pkl"),
            "prediction_solutions": self.get_luigi_local_target_with_task_id("prediction_solutions.pkl"),
        }

    def _get_solutions_and_objective_values(self, costs):
        self._init_optimizer()

        sols = []
        objs = []
        for c in tqdm(costs):
            self.optimizer.setObj(c)
            sol, obj = self.optimizer.solve()
            sols.append(sol)
            objs.append(obj)
        return np.array(sols), np.array(objs)

    def run(self):
        c_test = self.load_pickle(self.input()["split_dataset"]["c_test"].path)
        true_sols, true_objs = self._get_solutions_and_objective_values(c_test)
        self.dump_pickle(true_sols, self.output()["true_solutions"].path)
        self.dump_pickle(true_objs, self.output()["true_objective_value"].path)

        predictions = self.load_pickle(self.input()["predictions"]["predictions"].path)
        pred_sols, _ = self._get_solutions_and_objective_values(predictions)
        self.dump_pickle(pred_sols, self.output()["prediction_solutions"].path)


class OptimizeWithGurobi(OptimizationModel):
    abstract = False

    def _init_optimizer(self, grid_size=(5, 5)):
        self.optimizer = pyepo.model.grb.shortestPathModel(grid_size)


class Evaluation(BaseTaskClass):
    abstract = False
    solutions = ClsParameter(tpe=OptimizationModel.return_type())
    predictions = ClsParameter(tpe=MultiOutputRegressionModel.return_type())
    split_dataset = ClsParameter(tpe=GenerateAndSplitDataSet.return_type())

    summary = {}
    mse = 0
    regret = 0

    def requires(self):
        return {
            "solutions": self.solutions(),
            "predictions": self.predictions(),
            "split_dataset": self.split_dataset()
        }

    def run(self):
        true_solutions = self.load_pickle(self.input()["solutions"]["true_solutions"].path)
        true_objective_values = self.load_pickle(self.input()["solutions"]["true_objective_value"].path)
        prediction_solutions = self.load_pickle(self.input()["solutions"]["prediction_solutions"].path)
        predictions = self.load_pickle(self.input()["predictions"]["predictions"].path)
        c_test = self.load_pickle(self.input()["split_dataset"]["c_test"].path)

        self._compute_regret(prediction_solutions, c_test, true_objective_values)
        self._compute_mse(predictions, c_test)
        self._write_pipeline_steps()
        self._save_outputs()

    def _compute_regret(self, pred_sols, true_costs, true_objs):
        for i, sol in enumerate(pred_sols):
            cost = true_costs[i]
            true_obj = true_objs[i]

            loss = np.dot(sol, cost) - true_obj
            self.regret += loss

        self.regret /= abs(true_objs.sum() + 1e-3)
        self.summary["regret"] = self.regret

    def _compute_mse(self, predictions, true_costs):
        self.mse = ((predictions - true_costs) ** 2).mean()
        self.summary["mse"] = self.mse

    def _write_pipeline_steps(self):
        self.summary["optimizer"] = self.requires()["solutions"].task_family
        self.summary["regressor"] = self.requires()["predictions"].task_family

    def _save_outputs(self):
        self.dump_json(self.summary, self.output()["summary"].path)

    def output(self):
        return {
            "summary": self.get_luigi_local_target_with_task_id("summary.json")
        }


if __name__ == "__main__":

    target = Evaluation.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if not actual is None or actual == 0:
        max_results = actual

    validator = UniqueTaskPipelineValidator([OptimizationModel, MultiOutputRegressionModel])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=True, detailed_summary=True)
    else:
        print("No results!")
