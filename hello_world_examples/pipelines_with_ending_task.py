import pickle
from os.path import exists
from pathlib import Path
import seaborn as sns

import luigi
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import inhabitation_task
from cls_python import FiniteCombinatoryLogic, Subtypes
from inhabitation_task import RepoMeta, LuigiCombinator
from unique_task_pipeline_validator import UniqueTaskPipelineValidator


class LoadDiabetesData(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return luigi.LocalTarget("diabetes.pkl")

    def run(self):
        diabetes = load_diabetes()
        df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                          columns=diabetes['feature_names'] + ['target'])

        df.to_pickle(self.output().path)


class TrainTestSplit(luigi.Task, LuigiCombinator):
    abstract = False
    diabetes = inhabitation_task.ClsParameter(tpe=LoadDiabetesData.return_type())

    def output(self):
        return [
            luigi.LocalTarget("x_train.pkl"),
            luigi.LocalTarget("x_test.pkl"),
            luigi.LocalTarget("y_train.pkl"),
            luigi.LocalTarget("y_test.pkl")
        ]

    def requires(self):
        return self.diabetes()

    def run(self):
        data = pd.read_pickle(self.input().path)
        X = data.drop(["target"], axis="columns")
        y = data[["target"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        X_train.to_pickle(self.output()[0].path)
        X_test.to_pickle(self.output()[1].path)
        y_train.to_pickle(self.output()[2].path)
        y_test.to_pickle(self.output()[3].path)


class TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    splitted_data = inhabitation_task.ClsParameter(tpe=TrainTestSplit.return_type())

    def requires(self):
        return self.splitted_data()


class TrainLinearRegressionModel(TrainRegressionModel):
    abstract = False

    def output(self):
        return luigi.LocalTarget("linear_reg.pkl")

    def run(self):
        x_train = pd.read_pickle(self.input()[0].path)
        y_train = pd.read_pickle(self.input()[2].path)

        reg = LinearRegression()
        reg.fit(x_train, y_train)

        with open(self.output().path, "wb") as outfile:
            pickle.dump(reg, outfile)


class TrainLassoLarsModel(TrainRegressionModel):
    abstract = False

    def output(self):
        return luigi.LocalTarget("lasso_lars.pkl")

    def run(self):
        x_train = pd.read_pickle(self.input()[0].path)
        y_train = pd.read_pickle(self.input()[2].path)

        reg = LassoLars()
        reg.fit(x_train, y_train)

        with open(self.output().path, "wb") as outfile:
            pickle.dump(reg, outfile)


class TrainSupportVectorModel(TrainRegressionModel):
    abstract = False

    def output(self):
        return luigi.LocalTarget("support_vector.pkl")

    def run(self):
        x_train = pd.read_pickle(self.input()[0].path)
        y_train = pd.read_pickle(self.input()[2].path)

        reg = SVR()
        reg.fit(x_train, y_train)

        with open(self.output().path, "wb") as outfile:
            pickle.dump(reg, outfile)


class GenerateAndUpdateLeaderboard(luigi.Task, LuigiCombinator):
    abstract = False
    regressor = inhabitation_task.ClsParameter(tpe=TrainRegressionModel.return_type())
    splitted_data = inhabitation_task.ClsParameter(tpe=TrainTestSplit.return_type())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False
        self.leaderboard = None
        self.rmse = None
        self.mae = None
        self.r2 = None

    def complete(self):
        return self.done

    def requires(self):
        return [self.regressor(), self.splitted_data()]

    def _get_regressor_name(self):
        return Path(self.input()[0].path).stem

    def output(self):
        return luigi.LocalTarget("leaderboard.csv")

    def run(self):
        with open(self.input()[0].path, 'rb') as file:
            reg = pickle.load(file)

        x_test = pd.read_pickle(self.input()[1][1].path)
        y_test = pd.read_pickle(self.input()[1][3].path)
        y_pred = reg.predict(x_test).ravel()

        self._compute_metrics(y_test, y_pred)
        self._update_leaderboard()
        self.done = True

    def _compute_metrics(self, y_true, y_pred):
        self.rmse = round(mean_squared_error(y_true, y_pred, squared=False), 3)
        self.mae = round(mean_absolute_error(y_true, y_pred), 3)
        self.r2 = round(r2_score(y_true, y_pred), 3)

    def _update_leaderboard(self):

        if exists(self.output().path) is False:
            leaderboard = pd.DataFrame(columns=["regressor", "RMSE", "MAE", "R2"])
        else:
            leaderboard = pd.read_csv(self.output().path, index_col="index")

        if self._get_regressor_name() not in leaderboard["regressor"].values:

            leaderboard.loc[leaderboard.shape[0]] = [self._get_regressor_name(), self.rmse, self.mae, self.r2]
            leaderboard.to_csv(self.output().path, index_label="index")
        else:
            print("scores already exist for: ", self._get_regressor_name())


class VisualizeLeaderboard(luigi.Task):
    sort_by = luigi.ChoiceParameter(default="RMSE", choices=["RMSE", "MAE", "R2"])
    leaderboard_path = luigi.Parameter(default="leaderboard.csv")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = False

    # def requires(self):
    #     return GenerateAndUpdateLeaderboard(regressor=TrainRegressionModel(),
    #                                         splitted_data=TrainTestSplit())

    def complete(self):
        return self.done

    def run(self):
        lb = pd.read_csv(self.leaderboard_path)
        lb = lb.sort_values(by=self.sort_by, ascending=True)

        lb = lb.reset_index()
        lb = lb.rename(columns={
            "index": "Leaderboard Index",
            "RMSE": "Root Mean Squared Error",
            "MAE": "Mean Absolute Error",
            "R2": "Coefficient of Determination (R\u00b2)"
        })

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle(
            "\nLeaderboard Top {} Models Metrics\nSorted by:{}\n\n".format(len(lb), self.sort_by),
            x=0.05, ha="left")

        order = lb["Leaderboard Index"].values

        sns.barplot(x="Leaderboard Index", y="Root Mean Squared Error", data=lb, ax=axes[0], order=order)
        sns.barplot(x="Leaderboard Index", y="Mean Absolute Error", data=lb, ax=axes[1], order=order)
        sns.barplot(x="Leaderboard Index", y="Coefficient of Determination (R\u00b2)", data=lb, ax=axes[2], order=order)

        plt.tight_layout()
        plt.savefig(self.output().path)
        plt.show()
        plt.close(fig)

        self.done = True

    def output(self):
        return luigi.LocalTarget("leaderboard.png")


if __name__ == '__main__':

    target = GenerateAndUpdateLeaderboard.return_type()
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

    results = [t() for t in inhabitation_result.evaluated[0:max_results]]

    results.append(VisualizeLeaderboard())

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")

    # luigi.build([VisualizeLeaderboard()])
