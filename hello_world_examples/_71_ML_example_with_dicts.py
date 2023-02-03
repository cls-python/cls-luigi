import pickle

import luigi
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import inhabitation_task
from inhabitation_task import RepoMeta, LuigiCombinator
from cls_python import FiniteCombinatoryLogic, Subtypes
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LassoLars
from unique_task_pipeline_validator import UniqueTaskPipelineValidator


class LoadDiabetesData(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return {"diabetes_data": luigi.LocalTarget("diabetes.pkl")}

    def run(self):
        diabetes = load_diabetes()
        df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                          columns=diabetes['feature_names'] + ['target'])

        df.to_pickle(self.output()["diabetes_data"].path)


class TrainTestSplit(luigi.Task, LuigiCombinator):
    abstract = False
    diabetes = inhabitation_task.ClsParameter(tpe=LoadDiabetesData.return_type())

    def output(self):
        return {
            "x_train": luigi.LocalTarget("x_train.pkl"),
            "x_test": luigi.LocalTarget("x_test.pkl"),
            "y_train": luigi.LocalTarget("y_train.pkl"),
            "y_test": luigi.LocalTarget("y_test.pkl"),
        }

    def requires(self):
        return {"diabetes_data": self.diabetes()}

    def run(self):
        data = pd.read_pickle(self.input()["diabetes_data"]["diabetes_data"].path)
        X = data.drop(["target"], axis="columns")
        y = data[["target"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        X_train.to_pickle(self.output()["x_train"].path)
        X_test.to_pickle(self.output()["x_test"].path)
        y_train.to_pickle(self.output()["y_train"].path)
        y_test.to_pickle(self.output()["y_test"].path)


class FitTransformScaler(luigi.Task, LuigiCombinator):
    abstract = True
    splitted_data = inhabitation_task.ClsParameter(tpe=TrainTestSplit.return_type())

    def requires(self):
        return self.splitted_data()


class FitTransformMinMaxScaler(FitTransformScaler):
    abstract = False

    def output(self):
        return {
            "scaled_x_train": luigi.LocalTarget("minmax_scaled_x_train.pkl"),
            "scaled_x_test": luigi.LocalTarget("minmax_scaled_x_test.pkl"),
            "scaler": luigi.LocalTarget("minmax_scaler.pkl")
        }

    def run(self):
        x_train = pd.read_pickle(self.input()["x_train"].path)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        scaled_x_train = pd.DataFrame(scaler.transform(x_train),
                                      columns=scaler.feature_names_in_,
                                      index=x_train.index)
        scaled_x_train.to_pickle(self.output()["scaled_x_train"].path)

        x_test = pd.read_pickle(self.input()["x_test"].path)
        scaler.transform(x_test)
        scaled_x_test = pd.DataFrame(scaler.transform(x_test),
                                     columns=scaler.feature_names_in_,
                                     index=x_test.index)
        scaled_x_test.to_pickle(self.output()["scaled_x_test"].path)

        with open(self.output()["scaler"].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)


class FitTransformRobustScaler(FitTransformScaler):
    abstract = False

    def output(self):
        return {
            "scaled_x_train": luigi.LocalTarget("robust_scaled_x_train.pkl"),
            "scaled_x_test": luigi.LocalTarget("robust_scaled_x_test.pkl"),
            "scaler": luigi.LocalTarget("robust_scaler.pkl")
        }

    def run(self):
        x_train = pd.read_pickle(self.input()["x_train"].path)
        scaler = RobustScaler()
        scaler.fit(x_train)
        scaled_x_train = pd.DataFrame(scaler.transform(x_train),
                                      columns=scaler.feature_names_in_,
                                      index=x_train.index)
        scaled_x_train.to_pickle(self.output()["scaled_x_train"].path)

        x_test = pd.read_pickle(self.input()["x_test"].path)
        scaler.transform(x_test)
        scaled_x_test = pd.DataFrame(scaler.transform(x_test),
                                     columns=scaler.feature_names_in_,
                                     index=x_test.index)
        scaled_x_test.to_pickle(self.output()["scaled_x_test"].path)

        with open(self.output()["scaler"].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)


class TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    scaled_feats = inhabitation_task.ClsParameter(tpe=FitTransformScaler.return_type())
    splitted_data = inhabitation_task.ClsParameter(tpe=TrainTestSplit.return_type())

    def requires(self):
        return {"scaled_feats": self.scaled_feats(),
                "splitted_data": self.splitted_data()}

    def _get_variant_label(self):
        return Path(self.input()["scaled_feats"]["scaled_x_train"].path).stem


class TrainLinearRegressionModel(TrainRegressionModel):
    abstract = False

    def output(self):
        return {"model": luigi.LocalTarget("linear_reg" + "-" + self._get_variant_label() + ".pkl")}

    def run(self):
        scaled_x_train = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_train"].path)
        y_train = pd.read_pickle(self.input()["splitted_data"]["y_train"].path)

        reg = LinearRegression()
        reg.fit(scaled_x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)


class TrainLassoLarsModel(TrainRegressionModel):
    abstract = False

    def output(self):
        return {"model": luigi.LocalTarget("lasso_lars" + "-" + self._get_variant_label() + ".pkl")}

    def run(self):
        scaled_x_train = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_train"].path)
        y_train = pd.read_pickle(self.input()["splitted_data"]["y_train"].path)

        reg = LassoLars()
        reg.fit(scaled_x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)


class EvaluateRegressionModel(luigi.Task, LuigiCombinator):
    abstract = False
    regressor = inhabitation_task.ClsParameter(tpe=TrainRegressionModel.return_type())
    scaled_feats = inhabitation_task.ClsParameter(tpe=FitTransformScaler.return_type())
    splitted_data = inhabitation_task.ClsParameter(tpe=TrainTestSplit.return_type())

    def requires(self):
        return {
            "regressor": self.regressor(),
            "scaled_feats": self.scaled_feats(),
            "splitted_data": self.splitted_data()
        }

    def _get_variant_label(self):
        return Path(self.input()["regressor"]["model"].path).stem

    def output(self):
        return {
            "y_pred": luigi.LocalTarget("y_pred" + "-" + self._get_variant_label() + ".pkl")
        }

    def run(self):
        with open(self.input()["regressor"]["model"].path, 'rb') as file:
            reg = pickle.load(file)

        scaled_x_test = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_test"].path)
        y_test = pd.read_pickle(self.input()["splitted_data"]["y_test"].path)
        y_pred = pd.DataFrame()
        y_pred["y_pred"] = reg.predict(scaled_x_test).ravel()
        rmse = round(mean_squared_error(y_test, y_pred, squared=False), 3)

        print(self._get_variant_label())
        print("RMSE: {}".format(rmse))

        y_pred.to_pickle(self.output()["y_pred"].path)


if __name__ == '__main__':

    target = EvaluateRegressionModel.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
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

    validator = UniqueTaskPipelineValidator([FitTransformScaler])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    # results = [t() for t in inhabitation_result.evaluated[0:max_results]] # this is what we should NOT be using in this case :)

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
