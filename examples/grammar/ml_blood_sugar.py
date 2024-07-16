import pickle
import luigi
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LassoLars
from cls_luigi.grammar.encoding import ApplicativeTreeGrammarEncoder


class ABSTRACT_LoadData(luigi.Task, LuigiCombinator):
    abstract = True


class LoadData(ABSTRACT_LoadData):
    abstract = False

    def output(self):
        return luigi.LocalTarget("diabetes.pkl")

    def run(self):
        diabetes = load_diabetes()
        df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                          columns=diabetes['feature_names'] + ['target'])

        df.to_pickle(self.output().path)


class ABSTRACT_TrainTestSplit(luigi.Task, LuigiCombinator):
    abstract = True


class TrainTestSplit(ABSTRACT_TrainTestSplit):
    abstract = False
    diabetes = ClsParameter(tpe=ABSTRACT_LoadData.return_type())

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


class ABSTRACT_FitTransformScaler(luigi.Task, LuigiCombinator):
    abstract = True
    splitted_data = ClsParameter(tpe=ABSTRACT_TrainTestSplit.return_type())

    def requires(self):
        return self.splitted_data()


class FitTransformMinMaxScaler(ABSTRACT_FitTransformScaler):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget("minmax_scaled_x_train.pkl"),
            luigi.LocalTarget("minmax_scaled_x_test.pkl"),
            luigi.LocalTarget("minmax_scaler.pkl")
        ]

    def run(self):
        x_train = pd.read_pickle(self.input()[0].path)
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        scaled_x_train = pd.DataFrame(scaler.transform(x_train),
                                      columns=scaler.feature_names_in_,
                                      index=x_train.index)
        scaled_x_train.to_pickle(self.output()[0].path)

        x_test = pd.read_pickle(self.input()[1].path)
        scaler.transform(x_test)
        scaled_x_test = pd.DataFrame(scaler.transform(x_test),
                                     columns=scaler.feature_names_in_,
                                     index=x_test.index)
        scaled_x_test.to_pickle(self.output()[1].path)

        with open(self.output()[2].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)


class FitTransformRobustScaler(ABSTRACT_FitTransformScaler):
    abstract = False

    def output(self):
        return [
            luigi.LocalTarget("robust_scaled_x_train.pkl"),
            luigi.LocalTarget("robust_scaled_x_test.pkl"),
            luigi.LocalTarget("robust_scaler.pkl")
        ]

    def run(self):
        x_train = pd.read_pickle(self.input()[0].path)
        scaler = RobustScaler()
        scaler.fit(x_train)
        scaled_x_train = pd.DataFrame(scaler.transform(x_train),
                                      columns=scaler.feature_names_in_,
                                      index=x_train.index)
        scaled_x_train.to_pickle(self.output()[0].path)

        x_test = pd.read_pickle(self.input()[1].path)
        scaler.transform(x_test)
        scaled_x_test = pd.DataFrame(scaler.transform(x_test),
                                     columns=scaler.feature_names_in_,
                                     index=x_test.index)
        scaled_x_test.to_pickle(self.output()[1].path)

        with open(self.output()[2].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)


class ABSTRACT_TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    scaled_feats = ClsParameter(tpe=ABSTRACT_FitTransformScaler.return_type())
    target_values = ClsParameter(tpe=ABSTRACT_TrainTestSplit.return_type())

    def requires(self):
        return [self.scaled_feats(), self.target_values()]

    def _get_variant_label(self):
        return Path(self.input()[0][0].path).stem


class TrainLinearRegressionModel(ABSTRACT_TrainRegressionModel):
    abstract = False

    def output(self):
        return luigi.LocalTarget("linear_reg" + "-" + self._get_variant_label() + ".pkl")

    def run(self):
        x_train = pd.read_pickle(self.input()[0][0].path)
        y_train = pd.read_pickle(self.input()[1][2].path)

        reg = LinearRegression()
        reg.fit(x_train, y_train)

        with open(self.output().path, "wb") as outfile:
            pickle.dump(reg, outfile)


class TrainLassoLarsModel(ABSTRACT_TrainRegressionModel):
    abstract = False

    def output(self):
        return luigi.LocalTarget("lasso_lars" + "-" + self._get_variant_label() + ".pkl")

    def run(self):
        x_train = pd.read_pickle(self.input()[0][0].path)
        y_train = pd.read_pickle(self.input()[1][2].path)

        reg = LassoLars()
        reg.fit(x_train, y_train)

        with open(self.output().path, "wb") as outfile:
            pickle.dump(reg, outfile)


class ABSTRACT_EvaluateRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True


class EvaluateRegressionModel(ABSTRACT_EvaluateRegressionModel):
    abstract = False
    regressor = ClsParameter(tpe=ABSTRACT_TrainRegressionModel.return_type())
    scaled_feats = ClsParameter(tpe=ABSTRACT_FitTransformScaler.return_type())
    target_values = ClsParameter(tpe=ABSTRACT_TrainTestSplit.return_type())

    def requires(self):
        return [self.regressor(),
                self.scaled_feats(),
                self.target_values()]

    def _get_variant_label(self):
        return Path(self.input()[0].path).stem

    def output(self):
        return luigi.LocalTarget("y_pred" + "-" + self._get_variant_label() + ".pkl")

    def run(self):
        with open(self.input()[0].path, 'rb') as file:
            reg = pickle.load(file)

        x_test = pd.read_pickle(self.input()[1][1].path)
        y_test = pd.read_pickle(self.input()[2][3].path)
        y_pred = pd.DataFrame()
        y_pred["y_pred"] = reg.predict(x_test).ravel()
        rmse = round(mean_squared_error(y_test, y_pred, squared=False), 3)

        print(self._get_variant_label())
        print("RMSE: {}".format(rmse))

        y_pred.to_pickle(self.output().path)


if __name__ == '__main__':

    target = ABSTRACT_EvaluateRegressionModel.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    rules = inhabitation_result.rules

    encoded = ApplicativeTreeGrammarEncoder(rules).encode_into_tree_grammar()
    for k, v in encoded.items():
        print(k)
        print(v)
        print("===================\n")
