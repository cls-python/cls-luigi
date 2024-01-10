import luigi
import numpy as np
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from base_task import BaseTaskClass
from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
import pyepo

from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from global_parameters import GlobalParameters


class SyntheticDataGenerator(BaseTaskClass):
    abstract = False

    def run(self):
        features, costs = pyepo.data.shortestpath.genData(
            num_data=self.global_params.num_data + 1000,
            num_features=self.global_params.num_features,
            grid=self.global_params.grid,
            deg=self.global_params.deg,
            noise_width=self.global_params.noise_width,
            seed=self.global_params.seed
        )

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            costs,
            test_size=1000,
            random_state=self.global_params.seed
        )

        self.dump_pickle(x_train, self.output()["x_train"].path)
        self.dump_pickle(x_test, self.output()["x_test"].path)
        self.dump_pickle(y_train, self.output()["y_train"].path)
        self.dump_pickle(y_test, self.output()["y_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl"),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl")
        }


class MultiOutputRegressionModel(BaseTaskClass):
    abstract = True
    split_dataset = ClsParameter(tpe=SyntheticDataGenerator.return_type())

    regressor = None

    def requires(self):
        return self.split_dataset()

    def output(self):
        return {
            "test_predictions": self.get_luigi_local_target_with_task_id("test_predictions.pkl"),
            "fitted_model": self.get_luigi_local_target_with_task_id("fitted_model.pkl")
        }

    def _load_split_input_dataset(self):
        x_train = self.load_pickle(self.input()["x_train"].path)
        x_test = self.load_pickle(self.input()["x_test"].path)
        y_train = self.load_pickle(self.input()["y_train"].path)
        y_test = self.load_pickle(self.input()["y_test"].path)
        return x_train, x_test, y_train, y_test

    def run(self):
        x_train, x_test, y_train, y_test = self._load_split_input_dataset()

        self._init_regressor()
        self.regressor.fit(x_train, y_train)

        test_predictions = self.regressor.predict(x_test)
        self.dump_pickle(test_predictions, self.output()["test_predictions"].path)
        self.dump_pickle(self.regressor, self.output()["fitted_model"].path)

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
    split_dataset = ClsParameter(tpe=SyntheticDataGenerator.return_type())
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
            "optimal_solutions": self.get_luigi_local_target_with_task_id(
                "optimal_solutions.pkl"),
            "optimal_objective_values": self.get_luigi_local_target_with_task_id(
                "optimal_objective_value.pkl"),
            "test_prediction_solutions": self.get_luigi_local_target_with_task_id(
                "test_prediction_solutions.pkl"),
        }

    def _get_solutions_and_objective_values(self, costs):
        self._init_optimizer()

        sols = []
        objs = []
        for c in costs:
            self.optimizer.setObj(c)
            sol, obj = self.optimizer.solve()
            sols.append(sol)
            objs.append(obj)
        return np.array(sols), np.array(objs)

    def run(self):
        y_test = self.load_pickle(self.input()["split_dataset"]["y_test"].path)
        optimal_sols, optimal_objs = self._get_solutions_and_objective_values(y_test)
        self.dump_pickle(optimal_sols, self.output()["optimal_solutions"].path)
        self.dump_pickle(optimal_objs, self.output()["optimal_objective_values"].path)

        test_predictions = self.load_pickle(self.input()["predictions"]["test_predictions"].path)
        prediction_sols, _ = self._get_solutions_and_objective_values(test_predictions)
        self.dump_pickle(prediction_sols, self.output()["test_prediction_solutions"].path)


class OptimizeWithGurobi(OptimizationModel):
    abstract = False

    def _init_optimizer(self, grid_size=(5, 5)):
        self.optimizer = pyepo.model.grb.shortestPathModel(grid_size)


class Evaluation(BaseTaskClass):
    abstract = False
    sols_and_objs = ClsParameter(tpe=OptimizationModel.return_type())
    predictions = ClsParameter(tpe=MultiOutputRegressionModel.return_type())
    split_dataset = ClsParameter(tpe=SyntheticDataGenerator.return_type())

    summary = {}
    mse = None
    regret = None

    def requires(self):
        return {
            "sols_and_objs": self.sols_and_objs(),
            "predictions": self.predictions(),
            "split_dataset": self.split_dataset()
        }

    def run(self):
        true_objective_values = self.load_pickle(self.input()["sols_and_objs"]["optimal_objective_values"].path)
        optimal_solutions = self.load_pickle(self.input()["sols_and_objs"]["optimal_solutions"].path)
        prediction_solutions = self.load_pickle(self.input()["sols_and_objs"]["test_prediction_solutions"].path)
        predictions = self.load_pickle(self.input()["predictions"]["test_predictions"].path)
        y_test = self.load_pickle(self.input()["split_dataset"]["y_test"].path)

        self._compute_regret(prediction_solutions, y_test, true_objective_values, optimal_solutions)
        self._compute_mse(predictions, y_test)
        self._write_pipeline_steps()
        self._save_outputs()

    def _compute_regret(self, predicted_sols, true_costs, true_objs, optimal_solutions, minimization=True):
        self.regret = 0
        for index, predicted_sol in enumerate(predicted_sols):
            true_cost = true_costs[index]
            true_obj = true_objs[index]

            if minimization is True:
                _regret = np.dot(predicted_sol, true_cost) - true_obj

            elif minimization is False:
                _regret = true_obj - np.dot(predicted_sol, true_cost)

            if _regret < 0:
                if np.array_equal(predicted_sol, optimal_solutions[index]):
                    _regret = 0

            self.regret += _regret

        self.regret /= true_objs.sum()
        if self.regret < 0:
            print("regret is negative")

        self.summary["regret"] = self.regret

    def _compute_mse(self, predictions, true_costs):
        self.mse = ((predictions - true_costs) ** 2).mean()
        self.summary["mse"] = self.mse

    def _write_pipeline_steps(self):
        self.summary["optimizer"] = self.requires()["sols_and_objs"].task_family
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
        gp = GlobalParameters()
        for training_size in [100]:  # , 1000, 5000]:
            for deg in [1]:  # , 2, 3, 4, 5, 6]:
                for noice in [0]:  # , .5]:
                    for seed in [1]:  # , 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        gp.num_data = training_size
                        gp.deg = deg
                        gp.noise_width = noice
                        gp.seed = seed
                        gp.grid = (5, 5)
                        gp.num_features = 5
                        gp.dataset_name = "shortest_path-" + "ts_" + str(training_size) + "-deg_" + str(
                            deg) + "-noise_" + str(
                            noice) + "-seed_" + str(seed)

                        luigi.build(results, local_scheduler=True, detailed_summary=True)
    else:
        print("No results!")
