import json
import os
import pickle
import luigi
import networkx as nx
from cls.debug_util import deep_str
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from cls_luigi.inhabitation_task import RepoMeta, LuigiCombinator, ClsParameter, CLSLugiEncoder
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

import logging

from cls_luigi.search import UniqueActionFilter
from cls_luigi.tools.io_functions import dump_json


class GlobalPipelineParameters(luigi.Config):
    x_train_path = luigi.OptionalStrParameter(default=None)
    x_test_path = luigi.OptionalStrParameter(default=None)
    y_train_path = luigi.OptionalStrParameter(default=None)
    y_test_path = luigi.OptionalStrParameter(default=None)
    luigi_outputs_dir = luigi.OptionalStrParameter(default=None)
    pipelines_outputs_dir = luigi.OptionalStrParameter(default=None)

    n_jobs = luigi.IntParameter(default=None)
    seed = luigi.IntParameter(default=None)

    def set_parameters(self, dictionary):
        for k, v in dictionary.items():
            if hasattr(self, k):
                setattr(self, k, v)


class CLSLuigiBaseTask(luigi.Task, LuigiCombinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_params = GlobalPipelineParameters()
        self.logger = self.get_luigi_logger()

    @staticmethod
    def get_luigi_logger() -> logging.Logger:
        return logging.getLogger('luigi-root')

    def get_unique_output_path(self, file_name: str):
        return luigi.LocalTarget(
            pjoin(self.global_params.pipelines_outputs_dir, f"{self.task_id}-{file_name}")
        )


class LoadDataset(CLSLuigiBaseTask):
    abstract = True


class LoadPklDataset(LoadDataset):
    abstract = False

    def output(self):
        return {
            "x_train": self.get_unique_output_path("x_train.pkl"),
            "x_test": self.get_unique_output_path("x_test.pkl"),
            "y_train": self.get_unique_output_path("y_train.pkl"),
            "y_test": self.get_unique_output_path("y_test.pkl"),
        }

    def run(self):
        X_train = pd.read_pickle(self.global_params.x_train_path)
        X_test = pd.read_pickle(self.global_params.x_test_path)
        y_train = pd.read_pickle(self.global_params.y_train_path)
        y_test = pd.read_pickle(self.global_params.y_test_path)

        X_train.to_pickle(self.output()["x_train"].path)
        X_test.to_pickle(self.output()["x_test"].path)
        y_train.to_pickle(self.output()["y_train"].path)
        y_test.to_pickle(self.output()["y_test"].path)


class Scale(CLSLuigiBaseTask):
    abstract = True
    splitted_data = ClsParameter(tpe=LoadDataset.return_type())

    def requires(self):
        return self.splitted_data()


class MinMax(Scale):
    abstract = False

    def output(self):
        return {
            "scaled_x_train": self.get_unique_output_path("minmax_scaled_x_train.pkl"),
            "scaled_x_test": self.get_unique_output_path("minmax_scaled_x_test.pkl"),
            "scaler": self.get_unique_output_path("minmax_scaler.pkl")
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


class Robust(Scale):
    abstract = False

    def output(self):
        return {
            "scaled_x_train": self.get_unique_output_path("robust_scaled_x_train.pkl"),
            "scaled_x_test": self.get_unique_output_path("robust_scaled_x_test.pkl"),
            "scaler": self.get_unique_output_path("robust_scaler.pkl")
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


class Reg(CLSLuigiBaseTask):
    abstract = True
    scaled_feats = ClsParameter(tpe=Scale.return_type())
    target_values = ClsParameter(tpe=LoadDataset.return_type())

    def requires(self):
        return {"scaled_feats": self.scaled_feats(),
                "splitted_data": self.target_values()}

    def _get_variant_label(self):
        return Path(self.input()["scaled_feats"]["scaled_x_train"].path).stem


class LR(Reg):
    abstract = False

    def output(self):
        return {
            "model": self.get_unique_output_path("linear_reg.pkl")}

    def run(self):
        scaled_x_train = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_train"].path)
        y_train = pd.read_pickle(self.input()["splitted_data"]["y_train"].path)

        reg = LinearRegression()
        reg.fit(scaled_x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)


class RF(Reg):
    abstract = False

    def output(self):
        return {
            "model": self.get_unique_output_path("random_forest.pkl")}

    def run(self):
        scaled_x_train = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_train"].path)
        y_train = pd.read_pickle(self.input()["splitted_data"]["y_train"].path)

        reg = RandomForestRegressor(random_state=self.global_params.seed, n_jobs=self.global_params.n_jobs)
        reg.fit(scaled_x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)


class SlowRegressorEmulator(Reg):
    abstract = False

    def output(self):
        return {
            "model": self.get_unique_output_path("slow_reg.pkl")}

    def run(self):
        import time
        time.sleep(30)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump([1, 2, 3], outfile)


class FailedRegressorEmulator(Reg):
    abstract = False

    def output(self):
        return {
            "model": self.get_unique_output_path("failed_regressor.pkl")}

    def run(self):
        raise TypeError("Im the failing regressor")


class Eval(CLSLuigiBaseTask):
    abstract = True


class Evaluate(Eval):
    abstract = False
    regressor = ClsParameter(tpe=Reg.return_type())
    scaled_feats = ClsParameter(tpe=Scale.return_type())
    splitted_data = ClsParameter(tpe=LoadDataset.return_type())
    rmse = None

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
            "y_pred": self.get_unique_output_path("y_pred.pkl"),
            "score": self.get_unique_output_path("score.pkl")
        }

    def get_score(self, metric):
        if os.path.exists(self.output()["score"].path):
            with open(self.output()["score"].path, "rb") as f:
                score = pickle.load(f)

            return {"test": score, "train": score}
        raise FileNotFoundError("Score file not found!")

    def run(self):
        with open(self.input()["regressor"]["model"].path, 'rb') as file:
            reg = pickle.load(file)

        scaled_x_test = pd.read_pickle(self.input()["scaled_feats"]["scaled_x_test"].path)
        y_test = pd.read_pickle(self.input()["splitted_data"]["y_test"].path)
        y_pred = pd.DataFrame()
        y_pred["y_pred"] = reg.predict(scaled_x_test).ravel()
        self.rmse = float(mean_squared_error(y_test, y_pred, squared=False))
        # rmse = sklearn.metrics.explained_variance_score(y_test, y_pred)

        y_pred.to_pickle(self.output()["y_pred"].path)
        with open(self.output()["score"].path, "wb") as outfile:
            pickle.dump(self.rmse, outfile)


def download_and_split_diabetes_dataset(to_dir: str = "dataset/diabetes", seed: int = 42):
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    CWD = os.getcwd()

    diabetes = load_diabetes()
    df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                      columns=diabetes['feature_names'] + ['target'])

    X = df.drop(["target"], axis="columns")
    y = df[["target"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    os.makedirs(to_dir, exist_ok=True)
    X_train_path = pjoin(CWD, to_dir, "X_train.pkl")
    X_test_path = pjoin(CWD, to_dir, "X_test.pkl")
    y_train_path = pjoin(CWD, to_dir, "y_train.pkl")
    y_test_path = pjoin(CWD, to_dir, "y_test.pkl")

    X_train.to_pickle(X_train_path)
    X_test.to_pickle(X_test_path)
    y_train.to_pickle(y_train_path)
    y_test.to_pickle(y_test_path)

    return X_train_path, X_test_path, y_train_path, y_test_path


if __name__ == "__main__":
    import logging
    from cls_luigi.search.mcts import mcts_manager
    from cls_luigi.search.helpers import set_seed
    from cls_luigi.grammar import ApplicativeTreeGrammarEncoder
    from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, build_hypergraph, \
        render_hypergraph_components
    from cls_luigi.tools.constants import MINIMIZE

    from os.path import join as pjoin
    from os import makedirs, getcwd
    from cls.fcl import FiniteCombinatoryLogic
    from cls.subtypes import Subtypes

    logging.basicConfig(level=logging.DEBUG)

    SEED = 435
    N_JOBS = 1
    PIPELINE_METRIC = "root_mean_squared_error"
    SENSE = MINIMIZE
    MCTS_PARAMS = {
        "max_seconds": 10,
        "exploration_param": 2,
    }
    COMPONENT_TIMEOUT = None
    PIPELINE_TIMEOUT = 5
    PUNISHMENT_VALUE = 100
    FILTER = UniqueActionFilter()
    DS_NAME = "diabetes"
    CWD = getcwd()
    OUTPUTS_DIR = pjoin(CWD, DS_NAME)
    RUN_DIR = pjoin(OUTPUTS_DIR, f"seed-{SEED}")
    CLS_LUIGI_OUTPUTS_DIR = pjoin(RUN_DIR, "cls_luigi")
    CLS_LUIGI_PIPELINES_DIR = pjoin(CLS_LUIGI_OUTPUTS_DIR, "pipelines")
    LUIGI_OUTPUTS_DIR = pjoin(RUN_DIR, "luigi")
    LUIGI_PIPELINES_OUTPUTS_DIR = pjoin(LUIGI_OUTPUTS_DIR, "pipelines_outputs")

    set_seed(SEED)
    makedirs(OUTPUTS_DIR, exist_ok=True)
    makedirs(RUN_DIR, exist_ok=False)
    makedirs(CLS_LUIGI_OUTPUTS_DIR, exist_ok=False)
    makedirs(CLS_LUIGI_PIPELINES_DIR, exist_ok=False)
    makedirs(LUIGI_OUTPUTS_DIR, exist_ok=False)
    makedirs(LUIGI_PIPELINES_OUTPUTS_DIR, exist_ok=False)

    X_train_path, X_test_path, y_train_path, y_test_path = download_and_split_diabetes_dataset(seed=SEED)

    target_class = Eval
    target = target_class.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 200
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual
    pipelines_classes = [t for t in inhabitation_result.evaluated[0:max_results]]
    # pipelines = pipelines[0:1]
    for p in pipelines_classes:
        with open(pjoin(CLS_LUIGI_PIPELINES_DIR, f"{p().task_id}.json"), "w") as f:
            json.dump(p, f, cls=CLSLugiEncoder)

    rtg = inhabitation_result.rules
    with open(pjoin(CLS_LUIGI_OUTPUTS_DIR, "applicative_regular_tree_grammar.txt"), "w") as f:
        f.write(deep_str(rtg))

    tree_grammar = ApplicativeTreeGrammarEncoder(rtg, target_class.__name__).encode_into_tree_grammar()
    with open(pjoin(CLS_LUIGI_OUTPUTS_DIR, "regular_tree_grammar.json"), "w") as f:
        json.dump(tree_grammar, f, indent=4)

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    with open(pjoin(CLS_LUIGI_OUTPUTS_DIR, "grammar_nx_hypergraph.pkl"), "wb") as f:
        pickle.dump(hypergraph, f)

    nx.write_graphml(hypergraph, pjoin(CLS_LUIGI_OUTPUTS_DIR, "grammar_nx_hypergraph.graphml"))
    render_hypergraph_components(hypergraph, pjoin(CLS_LUIGI_OUTPUTS_DIR, "grammar_hypergraph.png"), node_size=5000,
                                 node_font_size=11)

    _luigi_pipeline_params = {
        "x_train_path": X_train_path,
        "x_test_path": X_test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
        "luigi_outputs_dir": LUIGI_OUTPUTS_DIR,
        "pipelines_outputs_dir": LUIGI_PIPELINES_OUTPUTS_DIR,
        "seed": SEED,
        "n_jobs": N_JOBS
    }
    luigi_pipeline_params = GlobalPipelineParameters().set_parameters(_luigi_pipeline_params)
    dump_json(pjoin(LUIGI_OUTPUTS_DIR, "luigi_pipeline_params.json"), _luigi_pipeline_params)

    pipeline_objects = [pipeline() for pipeline in pipelines_classes]
    mcts_manager = mcts_manager.MCTSManager(
        run_dir=RUN_DIR,
        pipeline_objects=pipeline_objects,
        mcts_params=MCTS_PARAMS,
        hypergraph=hypergraph,
        game_sense=SENSE,
        pipeline_metric=PIPELINE_METRIC,
        evaluator_punishment_value=PUNISHMENT_VALUE,
        pipeline_timeout=PIPELINE_TIMEOUT,
        component_timeout=COMPONENT_TIMEOUT,
        pipeline_filters=[FILTER],
    )

    inc = mcts_manager.run_mcts()
    mcts_manager.save_results()
