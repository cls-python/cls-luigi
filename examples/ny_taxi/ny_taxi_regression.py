import luigi
import pandas as pd
import json
import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

from cls_luigi.inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta, InhabitationTask, TaskState, states
from cls.fcl import FiniteCombinatoryLogic, Subtypes

from cls_luigi_read_tabular_data import WriteSetupJson, ReadTabularData
from cls.debug_util import deep_str

from cls_luigi.grammar import ApplicativeTreeGrammarEncoder

from os.path import join as pjoin

from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

output_dir = "output"

class WriteCSVRegressionSetupJson(WriteSetupJson):
    abstract = False

    def run(self):
        d = {
            "csv_file": "data/taxy_trips_ny_2016-06-01to03_3%sample.csv",
            "date_column": ['pickup_datetime', 'dropoff_datetime'],
            "target_column": 'trip_duration'
        }
        with open('data/setup.json', 'w') as f:
            json.dump(d, f)


class ReadTaxiData(ReadTabularData):
    abstract = False

    def run(self):
        setup = self._read_setup()
        taxi = pd.read_csv(setup["csv_file"], parse_dates=setup["date_column"])
        taxi.to_pickle('data/tabular_data.pkl')


class PreprocessTabularData(luigi.Task, LuigiCombinator):
    abstract = True
    tabular_data = ClsParameter(tpe=ReadTabularData.return_type())

    def requires(self):
        return self.tabular_data()

    def _read_tabular_data(self):
        return pd.read_pickle(self.input().open().name)


class PreprocessDateFeature1(PreprocessTabularData):
    abstract = False
    tabular_data = ClsParameter(tpe=ReadTabularData.return_type())

    def run(self):
        tabular = self._read_tabular_data()

        for c in tabular.columns:
            if pd.api.types.is_datetime64_any_dtype(tabular[c]):
                print("Datetime Column:", c)
                tabular[c + "_YEAR"] = tabular[c].dt.year
                tabular[c + "_MONTH"] = tabular[c].dt.hour
                tabular[c + "_DAY"] = tabular[c].dt.day


        print(tabular.columns)
        corr = tabular.corr().abs()

        # plt.figure(figsize=(10, 10))
        # sns.heatmap(corr, annot=True, cbar=False, fmt='.2f')
        # plt.tight_layout()
        # plt.savefig("data/corr_heatmap.png")
        tabular.to_pickle('data/tabular_data_preprocessed1.pkl')

    def output(self):
        return luigi.LocalTarget('data/tabular_data_preprocessed1.pkl')


class PreprocessDateFeature2(PreprocessTabularData):
    abstract = False
    tabular_data = ClsParameter(tpe=ReadTabularData.return_type())

    def run(self):
        tabular = self._read_tabular_data()

        for c in tabular.columns:
            if pd.api.types.is_datetime64_any_dtype(tabular[c]):
                print("Datetime Column:", c)
                tabular[c + "_WEEKDAY"] = tabular[c].dt.dayofweek
                tabular[c + "_HOUR"] = tabular[c].dt.hour

        tabular.to_pickle('data/tabular_data_preprocessed2.pkl')

    def output(self):
        return luigi.LocalTarget('data/tabular_data_preprocessed2.pkl')


class PreprocessorDummy(PreprocessDateFeature1, PreprocessDateFeature2):
    abstract = False
    tabular_data = ClsParameter(tpe=ReadTabularData.return_type())

    def run(self):
        tabular = self._read_tabular_data()
        tabular.to_pickle('data/tabular_data_preprocessed_dummy.pkl')

    def output(self):
        return luigi.LocalTarget('data/tabular_data_preprocessed_dummy.pkl')


d = {PreprocessDateFeature1, PreprocessDateFeature2}

class PreprocessorSet(luigi.Task, LuigiCombinator):
    date_feature1 = ClsParameter(tpe=PreprocessDateFeature1.return_type() \
        if PreprocessDateFeature1 in d else PreprocessorDummy.return_type())

    date_feature2 = ClsParameter(tpe=PreprocessDateFeature2.return_type() \
        if PreprocessDateFeature2 in d else PreprocessorDummy.return_type())

    def requires(self):
        return [self.date_feature1(), self.date_feature2()]

    def run(self):
        df = None
        for r in self.input():
            if df is not None:
                data = pd.read_pickle(r.open().name)
                df = pd.merge(df, data)
            else:
                df = pd.read_pickle(r.open().name)

        df.to_pickle('data/tabular_data_preprocessed.pkl')

    def output(self):
        return luigi.LocalTarget('data/tabular_data_preprocessed.pkl')


class TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    tabular_data_preprocessed = ClsParameter(tpe=PreprocessorSet.return_type())
    setup = ClsParameter(tpe=WriteSetupJson.return_type())

    def requires(self):
        return [self.setup(), self.tabular_data_preprocessed()]

    def output(self):
        return luigi.LocalTarget('data/regression_model.pkl')

    def _read_setup(self):
        with open(self.input()[0].open().name) as file:
            setup = json.load(file)
        return setup

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[1].open().name)


class TrainLinearRegressionModel(TrainRegressionModel):
    abstract = False

    def run(self):
        setup = self._read_setup()
        tabular = self._read_tabular_data()
        print("TARGET:", setup["target_column"])
        print("NOW WE FIT A REGRESSION MODEL")

        # Todo: Here I'm filtering to the non categorical columns!
        X = tabular[['trip_distance','trip_duration',
                    'haversine_distance', 'totalleft',
                    'totalsteps', 'totalturn', 'main_street_ratio',
                    'osrm_duration']]
        y = tabular[[setup["target_column"]]].values.ravel()
        print(y)
        print(X.shape)
        print(y.shape)
        reg = LinearRegression().fit(X, y)

        print(reg.coef_)

        with open('data/regression_model.pkl', 'wb') as f:
            pickle.dump(reg, f)


class FinalNode(luigi.WrapperTask, LuigiCombinator):
    setup = ClsParameter(tpe=WriteSetupJson.return_type())
    train = ClsParameter(tpe=TrainLinearRegressionModel.return_type())

    def requires(self):
        return [self.setup(), self.train()]


if __name__ == '__main__':
    import json
    import os
    RESULTUS_DIR = "results"
    
    os.makedirs(RESULTUS_DIR, exist_ok=True)
    from os.path import join as pjoin
    from os import makedirs, getcwd
    from llm_suggester.gemini import DirectGrammarAgent
    target_class = FinalNode
        
    repository = RepoMeta.repository
    
    DS_NAME = "output"
    CWD = getcwd()
    OUTPUTS_DIR = pjoin(CWD, DS_NAME)
    RUN_DIR = pjoin(OUTPUTS_DIR)
    CLS_LUIGI_OUTPUTS_DIR = pjoin(RUN_DIR, "cls_luigi")
    CLS_LUIGI_PIPELINES_DIR = pjoin(CLS_LUIGI_OUTPUTS_DIR, "pipelines")
    LUIGI_OUTPUTS_DIR = pjoin(RUN_DIR, "luigi")
    LUIGI_PIPELINES_OUTPUTS_DIR = pjoin(LUIGI_OUTPUTS_DIR, "pipelines_outputs")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)


    target = target_class.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    rtg = inhabitation_result.rules
    with open(pjoin(output_dir, "applicative_regular_tree_grammar.txt"), "w") as f:
        f.write(deep_str(rtg))
    tree_grammar = ApplicativeTreeGrammarEncoder(rtg, target_class.__name__).encode_into_tree_grammar()
    with open(pjoin(output_dir, "regular_tree_grammar.json"), "w") as f:
        json.dump(tree_grammar, f, indent=4)

    task = "A regression task to analyze the New York City traffic data based on a dataset of NYC taxi trips."

    agent = DirectGrammarAgent(task, tree_grammar, RUN_DIR)
    llm_suggested_grammar = agent.generate_reduced_grammar()
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0:max_results]]
    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        # with open(pjoin(output_dir, "regular_tree_grammar.json"), "w") as f:
        #     json.dump(tree_grammar, f, indent=4)
        print("Run Pipelines")
        luigi.build(results, local_scheduler=True, detailed_summary=True)  # für luigid: local_scheduler = True weglassen!
    else:
        print("No results!")
        
    

    task = InhabitationTask()
    states[task.task_id] = TaskState(fcl, target)
    luigi.build([task], worker_scheduler_factory=states[task.task_id].worker_scheduler_factory)