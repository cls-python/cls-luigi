from luigi import LocalTarget, build
from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.cls_tasks import ClsTask

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import json
import pickle
import os
from os.path import join as pjoin
from os import mkdir


DIR = os.path.dirname(__file__)
OUTPUTS_DIR = pjoin(DIR, "data")
INPUT_DATA_NAME = "AirPassengers.csv"
INPUT_DATA_PATH = pjoin(DIR, INPUT_DATA_NAME)
TARGET_NAME = "Actual"
TRAIN_TO_INDEX = 120
DATE_INDEX_COL = "Date"
SEED = 123


class MakeOutputDir(ClsTask):
    abstract = False

    def output(self):
        return LocalTarget(OUTPUTS_DIR)

    def run(self):
        mkdir(OUTPUTS_DIR)


class ReadTimeSeries(ClsTask):
    abstract = False
    out_dir = ClsParameter(tpe=MakeOutputDir.return_type())

    def requires(self):
        return self.out_dir()

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, "time_series.pkl"
            )
        )

    def run(self):
        time_series = pd.read_csv(INPUT_DATA_PATH, parse_dates=[DATE_INDEX_COL])
        time_series.set_index(DATE_INDEX_COL, inplace=True)
        time_series.index = pd.DatetimeIndex(
            time_series.index.values,
            freq=time_series.index.inferred_freq)

        time_series.to_pickle(self.output().path)


class SplitData(ClsTask):
    abstract = False
    time_series_data = ClsParameter(tpe=ReadTimeSeries.return_type())

    def requires(self):
        return self.time_series_data()

    def output(self):
        return {
            "train": LocalTarget(pjoin(OUTPUTS_DIR, f"{self.get_variant_filename()}_train.pkl")),
            "test": LocalTarget(pjoin(OUTPUTS_DIR, f"{self.get_variant_filename()}_test.pkl")),
        }

    def run(self):
        time_series = pd.read_pickle(self.input().path)
        train = time_series.iloc[:TRAIN_TO_INDEX]
        test = time_series.iloc[TRAIN_TO_INDEX:]

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)


class FitPredictExponentialSmoothingModel(ClsTask):
    abstract = True
    split_data = ClsParameter(tpe=SplitData.return_type())

    def requires(self):
        return self.split_data()

    def output(self):
        return {
            "fitted_values": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_fitted_values.pkl"
                )
            ),
            "prediction": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_prediction.pkl"
                )
            ),
            "model": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_model.pkl"
                )
            )
        }

    def _get_train_and_test_dfs(self):
        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)
        return train, test


class ExponentialSmoothing1stOrder(FitPredictExponentialSmoothingModel):
    abstract = False
    exp_smoothing_type = "1st Order"

    def run(self):
        train, test = self._get_train_and_test_dfs()
        es = ExponentialSmoothing(
            train[TARGET_NAME]
        ).fit()

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(es, outfile)

        fitted_values = pd.DataFrame(
            data=es.fittedvalues,
            columns=["Exp. Smoothing 1st Order"],
            index=train.index
        )

        prediction = pd.DataFrame(
            data=es.forecast(len(test)),
            columns=["Exp. Smoothing 1st Order"],
            index=test.index
        )

        fitted_values.to_pickle(self.output()["fitted_values"].path)
        prediction.to_pickle(self.output()["prediction"].path)


class ExponentialSmoothing2ndOrder(FitPredictExponentialSmoothingModel):
    abstract = False
    exp_smoothing_type = "2nd Order"

    def run(self):
        train, test = self._get_train_and_test_dfs()

        es = ExponentialSmoothing(
            train[TARGET_NAME],
            trend="add"
        ).fit()

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(es, outfile)

        fitted_values = pd.DataFrame(
            data=es.fittedvalues,
            columns=["Exp. Smoothing 2nd Order"],
            index=train.index
        )

        prediction = pd.DataFrame(
            data=es.forecast(len(test)),
            columns=["Exp. Smoothing 2nd Order"],
            index=test.index
        )

        fitted_values.to_pickle(self.output()["fitted_values"].path)
        prediction.to_pickle(self.output()["prediction"].path)


class ExponentialSmoothing3rdOrder(FitPredictExponentialSmoothingModel):
    abstract = False
    exp_smoothing_type = "3rd Order"

    def run(self):
        train, test = self._get_train_and_test_dfs()

        es = ExponentialSmoothing(
            train[TARGET_NAME],
            trend="mul",
            seasonal="mul"
        ).fit()

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(es, outfile)

        fitted_values = pd.DataFrame(
            data=es.fittedvalues,
            columns=["Exp. Smoothing 3rd Order"],
            index=train.index
        )

        prediction = pd.DataFrame(
            data=es.forecast(len(test)),
            columns=["Exp. Smoothing 3rd Order"],
            index=test.index
        )

        fitted_values.to_pickle(self.output()["fitted_values"].path)
        prediction.to_pickle(self.output()["prediction"].path)


class GenerateMonthAndYearColumns(ClsTask):
    abstract = False
    split_data = ClsParameter(tpe=SplitData.return_type())

    def requires(self):
        return self.split_data()

    def output(self):
        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_train.pkl"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_test.pkl"
                )
            )
        }

    def run(self):
        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)

        train["Month"] = train.index.map(self._get_month)
        test["Month"] = test.index.map(self._get_month)

        train["Year"] = train.index.map(self._get_year)
        test["Year"] = test.index.map(self._get_year)

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)

    @staticmethod
    def _get_month(time_stamp):
        return time_stamp.month

    @staticmethod
    def _get_year(time_stamp):
        return time_stamp.year


class ToSupervisedLearningDataset(ClsTask):
    abstract = False
    split_data = ClsParameter(tpe=GenerateMonthAndYearColumns.return_type())

    n_in = 6

    def requires(self):
        return self.split_data()

    def output(self):
        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_train.pkl"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_test.pkl"
                )
            )
        }

    def run(self):
        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)

        concatenated = pd.concat([train, test])
        supervised_data = pd.DataFrame(
            columns=[f"t-{i}" for i in range(self.n_in, 0, -1)],
            data=self._get_supervised_learning_features_from_target(concatenated),
            index=concatenated.index[self.n_in:]
        )

        supervised_data = supervised_data.join(concatenated)
        supervised_train = supervised_data.iloc[: TRAIN_TO_INDEX - self.n_in]
        supervised_test = supervised_data.iloc[TRAIN_TO_INDEX - self.n_in:]

        supervised_train.to_pickle(self.output()["train"].path)
        supervised_test.to_pickle(self.output()["test"].path)

    def _get_supervised_learning_features_from_target(self, df):
        cols = []

        for i in range(self.n_in, 0, -1):
            cols.append(df[TARGET_NAME].shift(i))

        supervised_data = pd.concat(cols, axis=1)
        supervised_data.dropna(inplace=True)

        return supervised_data.values


class ExponentialSmoothingPreprocessor(ClsTask):
    abstract = False
    supervised_data = ClsParameter(tpe=ToSupervisedLearningDataset.return_type())
    expo_smoothing = ClsParameter(tpe=FitPredictExponentialSmoothingModel.return_type())

    def requires(self):
        return {
            "supervised_data": self.supervised_data(),
            "exp_smoothing": self.expo_smoothing()
        }

    def output(self):
        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_train.pkl"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_test.pkl"
                )
            )
        }

    def run(self):
        supervised_train = pd.read_pickle(self.input()["supervised_data"]["train"].path)
        supervised_test = pd.read_pickle(self.input()["supervised_data"]["test"].path)

        fitted_values = pd.read_pickle(self.input()["exp_smoothing"]["fitted_values"].path)
        es_prediction = pd.read_pickle(self.input()["exp_smoothing"]["prediction"].path)

        supervised_train = supervised_train.join(fitted_values)
        supervised_test = supervised_test.join(es_prediction)

        supervised_train.to_pickle(self.output()["train"].path)
        supervised_test.to_pickle(self.output()["test"].path)


class FitPredictRegressionModel(ClsTask):
    abstract = True

    def output(self):
        return {
            "model": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_model.pkl"
                )
            ),
            "prediction": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self.get_variant_filename()}_prediction.pkl"
                )
            )
        }

    def _get_split_train_test_data(self):
        train = pd.read_pickle(self.input()["data"]["train"].path)
        test = pd.read_pickle(self.input()["data"]["test"].path)

        x_train = train.drop(TARGET_NAME, axis=1)
        x_test = test.drop(TARGET_NAME, axis=1)
        y_train = train[TARGET_NAME]
        y_test = test[TARGET_NAME]

        return x_train, x_test, y_train, y_test

    def _get_exp_smoothing_type(self):
        requires = self.requires()
        exp_smoothing = requires["data"].requires()["exp_smoothing"]

        assert isinstance(exp_smoothing, FitPredictExponentialSmoothingModel)

        return exp_smoothing.exp_smoothing_type

    def _load_config(self):
        with open(self.input()["config"].path, "r") as infile:
            return json.load(infile)

    def _get_config_name(self):
        return self.requires()["config"].name


class LassoConfigs(ClsTask):
    abstract = True


class LassoConfig1(LassoConfigs):
    abstract = False
    name = "Lasso Config 1"

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, "lasso_config_1.json"
            )
        )

    def run(self):
        params = {
            "alpha": 0.5,
            "max_iter": 1000,
            "tol": 1e-4,
            "selection": "cyclic",
            "random_state": SEED
        }

        with open(self.output().path, "w+") as outfile:
            json.dump(params, outfile)


class LassoConfig2(LassoConfigs):
    abstract = False
    name = "Lasso Config 2"

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, "lasso_config_2.json"
            )
        )

    def run(self):
        params = {
            "alpha": 0.8,
            "max_iter": 2000,
            "tol": 1e-5,
            "selection": "random",
            "random_state": SEED
        }

        with open(self.output().path, "w+") as outfile:
            json.dump(params, outfile)


class FitPredictLasso(FitPredictRegressionModel):
    abstract = False
    supervised_split_data = ClsParameter(
        tpe={
            "without_exp_smoothing": ToSupervisedLearningDataset.return_type(),
            "with_exp_smoothing": ExponentialSmoothingPreprocessor.return_type()
        }
    )
    config = ClsParameter(tpe=LassoConfigs.return_type())

    def requires(self):
        return {
            "data": self.supervised_split_data(),
            "config": self.config()
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()
        config = self._load_config()
        config_name = self._get_config_name()

        lasso_reg = Lasso(**config).fit(x_train, y_train)
        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(lasso_reg, outfile)

        if self.config_index == "without_exp_smoothing":
            pred_col_name = "Linear Regression Prediction Without Exponential Smoothing Features"
        elif self.config_index == "with_exp_smoothing":
            exp_smoothing_type = self._get_exp_smoothing_type()
            pred_col_name = f"Linear Regression Prediction With Exponential Smoothing {exp_smoothing_type} Features"

        pred_col_name = pred_col_name + " " + config_name

        prediction = pd.DataFrame(
            data=lasso_reg.predict(x_test),
            index=x_test.index,
            columns=[pred_col_name]
        )
        prediction.to_pickle(self.output()["prediction"].path)


class RandomForestConfigs(ClsTask):
    abstract = True


class RandomForestConfig1(RandomForestConfigs):
    abstract = False
    name = "Random Forest Config 1"

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, "random_forest_config_1.json"
            )
        )

    def run(self):
        params = {
            "n_estimators": 200,
            "criterion": "absolute_error",
            "max_features": "log2",
            "random_state": SEED
        }

        with open(self.output().path, "w+") as outfile:
            json.dump(params, outfile)


class RandomForestConfig2(RandomForestConfigs):
    abstract = False
    name = "Random Forest Config 2"

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, "random_forest_config_2.json"
            )
        )

    def run(self):
        params = {
            "n_estimators": 150,
            "criterion": "squared_error",
            "max_features": "sqrt",
            "random_state": SEED
        }

        with open(self.output().path, "w+") as outfile:
            json.dump(params, outfile)


class FitPredictRandomForest(FitPredictRegressionModel):
    abstract = False
    supervised_split_data = ClsParameter(
        tpe={
            "without_exp_smoothing": ToSupervisedLearningDataset.return_type(),
            "with_exp_smoothing": ExponentialSmoothingPreprocessor.return_type()
        }
    )
    config = ClsParameter(tpe=RandomForestConfigs.return_type())

    def requires(self):
        return {
            "data": self.supervised_split_data(),
            "config": self.config()
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()
        config = self._load_config()
        config_name = self._get_config_name()

        random_forest_reg = RandomForestRegressor(**config).fit(x_train, y_train)
        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(random_forest_reg, outfile)

        if self.config_index == "without_exp_smoothing":
            pred_col_name = "Random Forest Prediction Without Exponential Smoothing Features"
        elif self.config_index == "with_exp_smoothing":
            exp_smoothing_type = self._get_exp_smoothing_type()
            pred_col_name = f"Random Forest Prediction With Exponential Smoothing {exp_smoothing_type} Features"

        pred_col_name = pred_col_name + " " + config_name

        prediction = pd.DataFrame(
            data=random_forest_reg.predict(x_test),
            index=x_test.index,
            columns=[pred_col_name]
        )
        prediction.to_pickle(self.output()["prediction"].path)


class ScoreAndVisualizePredictions(ClsTask):
    abstract = False

    dataset = ClsParameter(tpe=SplitData.return_type())
    predictions = ClsParameter(
        tpe={
            "ml_model": FitPredictRegressionModel.return_type(),
            "exp_smoothing": FitPredictExponentialSmoothingModel.return_type()
        }
    )

    config_domain = {"ml_model", "exp_smoothing"}

    rmse = None
    mae = None

    def requires(self):
        return {
            "dataset": self.dataset(),
            "predictions": self.predictions()
        }

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, f"{self.get_variant_filename()}_visual.png"
            )
        )

    def run(self):
        train = pd.read_pickle(self.input()["dataset"]["train"].path)
        test = pd.read_pickle(self.input()["dataset"]["test"].path)

        y_train = train[[TARGET_NAME]]
        y_test = test[[TARGET_NAME]]

        predictions = pd.read_pickle(self.input()["predictions"]["prediction"].path)

        self._compute_metrics(y_test, predictions)
        self._plot_and_save(y_train, y_test, predictions)

    def _plot_and_save(self, y_train, y_test, predictions):
        plt.figure(figsize=(16, 8))

        sns.lineplot(data=y_train[TARGET_NAME], marker="o", color="black", label="Train")
        sns.lineplot(data=y_test[TARGET_NAME], marker="o", color="orange", label="Test")

        pred_col_name = list(filter(lambda x: x != "Date", predictions))
        pred_col_name = pred_col_name[0]

        sns.lineplot(data=predictions[pred_col_name], marker="o", color="green", label=pred_col_name)

        plt.legend()
        plt.title(f"Prediction Visualization\n\nRMSE: {self.rmse:.5f}\nMAE: {self.mae:.5f}", loc="left")

        plt.tight_layout()
        plt.savefig(self.output().path)

    def _compute_metrics(self, y_test, predictions):
        self.rmse = mean_squared_error(y_test, predictions, squared=False)
        self.mae = mean_absolute_error(y_test, predictions)


if __name__ == "__main__":
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = ScoreAndVisualizePredictions.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    StaticJSONRepo(RepoMeta).dump_static_repo_json()

    # print(deep_str(inhabitation_result.rules))
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    validator = UniqueTaskPipelineValidator([FitPredictExponentialSmoothingModel, FitPredictRegressionModel])

    if not actual is None or actual == 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()

        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
