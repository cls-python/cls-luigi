import logging
import pickle

import luigi
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from cls_luigi.grammar import ApplicativeTreeGrammarEncoder
from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, build_hypergraph
from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LassoLars

from cls_luigi.search.mcts.evaluator import Evaluator
from cls_luigi.search.mcts.game import HyperGraphGame
from cls_luigi.search.mcts.policy import UCT
from cls_luigi.search.mcts.sp_mcts import SP_MCTS
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

RESULTUS_DIR = "results"



class Data(luigi.Task, LuigiCombinator):
    abstract = True

class data(luigi.Task, LuigiCombinator):
    abstract = False

    def output(self):
        return {"diabetes_data": luigi.LocalTarget(RESULTUS_DIR + "/" + "diabetes.pkl")}

    def run(self):
        diabetes = load_diabetes()
        df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                          columns=diabetes['feature_names'] + ['target'])

        df.to_pickle(self.output()["diabetes_data"].path)

class Split(luigi.Task, LuigiCombinator):
    abstract = True

class tts(luigi.Task, LuigiCombinator):
    abstract = False
    diabetes = ClsParameter(tpe=Data.return_type())

    def output(self):
        return {
            "x_train": luigi.LocalTarget(RESULTUS_DIR + "/" + "x_train.pkl"),
            "x_test": luigi.LocalTarget(RESULTUS_DIR + "/" + "x_test.pkl"),
            "y_train": luigi.LocalTarget(RESULTUS_DIR + "/" + "y_train.pkl"),
            "y_test": luigi.LocalTarget(RESULTUS_DIR + "/" + "y_test.pkl"),
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


class Scaler(luigi.Task, LuigiCombinator):
    abstract = True
    splitted_data = ClsParameter(tpe=Split.return_type())

    def requires(self):
        return self.splitted_data()


class minmax(Scaler):
    abstract = False

    def output(self):
        return {
            "scaled_x_train": luigi.LocalTarget(RESULTUS_DIR + "/" + "minmax_scaled_x_train.pkl"),
            "scaled_x_test": luigi.LocalTarget(RESULTUS_DIR + "/" + "minmax_scaled_x_test.pkl"),
            "scaler": luigi.LocalTarget(RESULTUS_DIR + "/" + "minmax_scaler.pkl")
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


class robust(Scaler):
    abstract = False

    def output(self):
        return {
            "scaled_x_train": luigi.LocalTarget(RESULTUS_DIR + "/" + "robust_scaled_x_train.pkl"),
            "scaled_x_test": luigi.LocalTarget(RESULTUS_DIR + "/" + "robust_scaled_x_test.pkl"),
            "scaler": luigi.LocalTarget(RESULTUS_DIR + "/" + "robust_scaler.pkl")
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


class Reg(luigi.Task, LuigiCombinator):
    abstract = True
    scaled_feats = ClsParameter(tpe=Scaler.return_type())
    splitted_data = ClsParameter(tpe=Split.return_type())

    def requires(self):
        return {"scaled_feats": self.scaled_feats(),
                "splitted_data": self.splitted_data()}

    def _get_variant_label(self):
        return Path(self.input()["scaled_feats"]["scaled_x_train"].path).stem


class lr(Reg):
    abstract = False

    def output(self):
        return {"model": luigi.LocalTarget(RESULTUS_DIR + "/" + "linear_reg" + "-" + self._get_variant_label() + ".pkl")}

    def run(self):
        scaled_x_train = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_train"].path)
        y_train = pd.read_pickle(self.input()["splitted_data"]["y_train"].path)

        reg = LinearRegression()
        reg.fit(scaled_x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)


class ll(Reg):
    abstract = False

    def output(self):
        return {"model": luigi.LocalTarget(RESULTUS_DIR + "/" + "lasso_lars" + "-" + self._get_variant_label() + ".pkl")}

    def run(self):
        scaled_x_train = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_train"].path)
        y_train = pd.read_pickle(self.input()["splitted_data"]["y_train"].path)

        reg = LassoLars()
        reg.fit(scaled_x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)

class Eval(luigi.Task, LuigiCombinator):
    abstract = True
class eval(Eval):
    abstract = False
    regressor = ClsParameter(tpe=Reg.return_type())
    scaled_feats = ClsParameter(tpe=Scaler.return_type())
    splitted_data = ClsParameter(tpe=Split.return_type())

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
            "y_pred": luigi.LocalTarget(RESULTUS_DIR + "/" + "y_pred" + "-" + self._get_variant_label() + ".pkl")
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


if __name__ == "__main__":
    import os
    os.makedirs(RESULTUS_DIR, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG)
    target_class = Eval

    target = target_class.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)

    inhabitation_result = fcl.inhabit(target)
    rules = inhabitation_result.rules

    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    results = [t() for t in inhabitation_result.evaluated[0:max_results]]


    tree_grammar = ApplicativeTreeGrammarEncoder(rules, target_class.__name__).encode_into_tree_grammar()

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    # plot_hypergraph_components(hypergraph, "binary_clf.png", start_node="CLF", node_size=5000, node_font_size=11)

    params = {
        "num_iterations": 500,
        "exploration_param": 0.5,
        "num_simulations": 1,
    }
    evaluator = Evaluator(pipelines=results)
    evaluator.populate_pipeline_map()
    game = HyperGraphGame(hypergraph, evaluator=evaluator)

    mcts = SP_MCTS(
        game=game,
        parameters=params,
        selection_policy=UCT,
    )
    mcts.run()
    mcts.draw_tree("nx_di_graph.png", plot=True)
    mcts.shut_down("mcts.pkl", "nx_di_graph.pkl")
