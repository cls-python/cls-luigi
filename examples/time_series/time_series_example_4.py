import pickle
import luigi
from luigi import LocalTarget
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
from cls.subtypes import Subtypes
from cls.fcl import FiniteCombinatoryLogic
from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo
from cls_luigi.cls_tasks import (ClsTask)
from cls_luigi.sub_pipeline_sequence_validator import SubPipelineSequenceValidator
import pandas as pd
import seaborn as sns

from cls_luigi.util.util import save_deps_tree_with_task_id

INPUT_DATA_NAME = "AirPassengers.csv"  # time series to read
TARGET_NAME = "Actual"
TRAIN_TO_INDEX = 120  # used for splitting the time series into train and test sets
FORCAST_PERIOD = 24  # number of periods to predict
DATE_INDEX_COL = "Date"  # name of the column containing the date
SEED = 123  # seed for reproducibility


class ReadTS(ClsTask):
    abstract = False

    def output(self):
        return LocalTarget(
            self.get_unique_output_path(file_name="time_series.pkl")
        )

    def run(self):
        time_series = pd.read_csv(INPUT_DATA_NAME, parse_dates=[DATE_INDEX_COL])
        time_series.set_index(DATE_INDEX_COL, inplace=True)
        time_series.index = pd.DatetimeIndex(
            time_series.index.values,
            freq=time_series.index.inferred_freq,
            name="index"
        )

        time_series.to_pickle(self.output().path)


class Tabular(ClsTask):
    abstract = True


class SimpleSplitTS(Tabular):
    abstract = False
    time_series_data = ClsParameter(tpe=ReadTS.return_type())

    def requires(self):
        return self.time_series_data()

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl'))
        }

    def run(self):
        time_series = pd.read_pickle(self.input().path)
        train = time_series.iloc[:TRAIN_TO_INDEX]
        test = time_series.iloc[TRAIN_TO_INDEX:]

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)


class TSToTabular(Tabular):
    abstract = True
    tabular = ClsParameter(tpe=Tabular.return_type())

    def requires(self):
        return self.tabular()

    def get_train_test_dfs(self, input_=None):
        if input_:
            train = pd.read_pickle(input_["train"].path)
            test = pd.read_pickle(input_["test"].path)
            return train, test

        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)
        return train, test


class ConfigAddLag(ClsTask):
    abstract = True

    def _dump_pickle_config(self, obj):
        with open(self.output().path, "wb") as outfile:
            pickle.dump(obj, outfile)


class ConfigAddLag7(ConfigAddLag):
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path('lag_7.pkl'))

    def run(self):
        lag = 7
        self._dump_pickle_config(lag)


class ConfigAddLag14(ConfigAddLag):
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path('lag_14.pkl'))

    def run(self):
        lag = 14
        self._dump_pickle_config(lag)


class AddLagColumn(TSToTabular):
    abstract = False

    tabular = ClsParameter(tpe=Tabular.return_type())
    config = ClsParameter(tpe=ConfigAddLag.return_type())

    def requires(self):
        return {
            "data": self.tabular(),
            "config": self.config()
        }

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl'))
        }

    def run(self):
        train, test = self.get_train_test_dfs(self.input()["data"])
        lag_period = self.get_lag_period()

        concatenated = pd.concat([train, test])
        supervised_data = pd.DataFrame(
            columns=[f"t-{i}" for i in range(lag_period, 0, -1)],
            data=self._get_supervised_learning_features_from_target(concatenated, lag_period),
            index=concatenated.index[lag_period:]
        )

        supervised_data = supervised_data.join(concatenated)
        supervised_train = supervised_data.iloc[: TRAIN_TO_INDEX - lag_period]
        supervised_test = supervised_data.iloc[TRAIN_TO_INDEX - lag_period:]

        supervised_train.to_pickle(self.output()["train"].path)
        supervised_test.to_pickle(self.output()["test"].path)

    def _get_supervised_learning_features_from_target(self, df, lag_period):
        cols = []

        for i in range(lag_period, 0, -1):
            cols.append(df[TARGET_NAME].shift(i))

        supervised_data = pd.concat(cols, axis=1)
        supervised_data.dropna(inplace=True)

        return supervised_data.values

    def get_lag_period(self):
        with open(self.input()["config"].path, "rb") as infile:
            lag = pickle.load(infile)
        return lag


class AddMonthYearColumn(TSToTabular):
    abstract = False

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl'))
        }

    def run(self):
        train, test = self.get_train_test_dfs()

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


class AddExponentialSmoothing2ndOrderColumn(TSToTabular):
    abstract = False

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl')),
            "model": LocalTarget(self.get_unique_output_path("model.pkl"))
        }

    def run(self):
        train, test = self.get_train_test_dfs()

        es = ExponentialSmoothing(
            train[TARGET_NAME],
            trend="add"
        ).fit()

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(es, outfile)

        train["Exp. Smoothing 2nd Order"] = es.fittedvalues
        test["Exp. Smoothing 2nd Order"] = es.forecast(FORCAST_PERIOD)

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)


class AddExponentialSmoothing3rdOrderColumn(TSToTabular):
    abstract = False

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl')),
            "model": LocalTarget(self.get_unique_output_path("model.pkl"))
        }

    def run(self):
        train, test = self.get_train_test_dfs()

        es = ExponentialSmoothing(
            train[TARGET_NAME],
            trend="mul",
            seasonal="mul",
        ).fit()

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(es, outfile)

        train["Exp. Smoothing 3rd Order"] = es.fittedvalues
        test["Exp. Smoothing 3rd Order"] = es.forecast(24)

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)


class TabularTSPredictor(ClsTask):
    abstract = True

    def output(self):
        return {
            "model": LocalTarget(self.get_unique_output_path("model.pkl")),
            "prediction": LocalTarget(self.get_unique_output_path("predictions.pkl")),
        }

    def _get_split_data(self):
        train = pd.read_pickle(self.input()["data"]["train"].path)
        test = pd.read_pickle(self.input()["data"]["test"].path)

        x_train = train.drop(TARGET_NAME, axis=1)
        x_test = test.drop(TARGET_NAME, axis=1)
        y_train = train[TARGET_NAME]
        y_test = test[TARGET_NAME]

        return x_train, x_test, y_train, y_test

    def _get_config(self):
        with open(self.input()["config"].path, "rb") as infile:
            return pickle.load(infile)


class ConfigRandomForest(ClsTask):
    abstract = True

    def _dump_pickle_config(self, params):
        with open(self.output().path, "wb") as outfile:
            pickle.dump(params, outfile)


class ConfigRandomForest1(ConfigRandomForest):
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path("rf_config1.pkl"))

    def run(self):
        params = {
            "n_estimators": 100,
            "criterion": "absolute_error",
            "max_features": "sqrt",
            "random_state": 42,
        }
        self._dump_pickle_config(params)


class ConfigRandomForest2(ConfigRandomForest):
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path("rf_config2.pkl"))

    def run(self):
        params = {
            "n_estimators": 50,
            "criterion": "squared_error",
            "max_features": "log2",
            "random_state": 42,
        }
        self._dump_pickle_config(params)


class FitPredictRandomForest(TabularTSPredictor):
    abstract = False

    tabular_train_test = ClsParameter(tpe=TSToTabular.return_type())
    config = ClsParameter(tpe=ConfigRandomForest.return_type())

    def requires(self):
        return {
            "data": self.tabular_train_test(),
            "config": self.config()
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_data()
        config = self._get_config()

        rf = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            criterion=config["criterion"],
            max_features=config["max_features"],
            random_state=config["random_state"]
        )

        rf.fit(x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(rf, outfile)

        predictions = pd.DataFrame(
            data=rf.predict(x_test),
            index=x_test.index,
            columns=["Random Forest Regression"]
        )

        predictions.to_pickle(self.output()["prediction"].path)


class ConfigLinearRegression(ClsTask):
    abstract = True

    def _dump_pickle_config(self, params):
        with open(self.output().path, "wb") as outfile:
            pickle.dump(params, outfile)


class ConfigLinearRegression1(ConfigLinearRegression):
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path("lr_config1.pkl"))

    def run(self):
        params = {
            "fit_intercept": False,
            "copy_X": False
        }
        self._dump_pickle_config(params)


class ConfigLinearRegression2(ConfigLinearRegression):
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path("lr_config2.pkl"))

    def run(self):
        params = {
            "fit_intercept": True,
            "copy_X": True
        }
        self._dump_pickle_config(params)


class FitPredictLinearRegression(TabularTSPredictor):
    abstract = False

    tabular_train_test = ClsParameter(tpe=TSToTabular.return_type())
    config = ClsParameter(tpe=ConfigLinearRegression.return_type())

    def requires(self):
        return {
            "data": self.tabular_train_test(),
            "config": self.config()
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_data()
        config = self._get_config()

        rf = LinearRegression(
            fit_intercept=config["fit_intercept"],
            copy_X=config["copy_X"],

        )

        rf.fit(x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(rf, outfile)

        predictions = pd.DataFrame(
            data=rf.predict(x_test),
            index=x_test.index,
            columns=["Linear Regression Regression"]
        )

        predictions.to_pickle(self.output()["prediction"].path)


class UnivariateTSPredictor(ClsTask):
    abstract = True
    split_data = ClsParameter(tpe=SimpleSplitTS.return_type())

    def requires(self):
        return self.split_data()

    def output(self):
        return {
            "fitted_values": LocalTarget(self.get_unique_output_path('fitted_values.pkl')),
            "prediction": LocalTarget(self.get_unique_output_path('prediction.pkl')),
            "model": LocalTarget(self.get_unique_output_path('model.pkl'))
        }

    def _get_train_and_test_dfs(self):
        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)
        return train, test


class ExponentialSmoothing1stOrder(UnivariateTSPredictor):
    abstract = False

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


class ExponentialSmoothing2ndOrder(UnivariateTSPredictor):
    abstract = False

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


class ExponentialSmoothing3rdOrder(UnivariateTSPredictor):
    abstract = False

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


class ScoreAndVisualizePredictions(ClsTask):
    abstract = True
    rmse = None
    mae = None

    def output(self):
        return LocalTarget(self.get_unique_output_path('visual.json.png'))

    def run(self):
        train = pd.read_pickle(self.input()["split_dataset"]["train"].path)
        test = pd.read_pickle(self.input()["split_dataset"]["test"].path)
        y_train = train[[TARGET_NAME]]
        y_test = test[[TARGET_NAME]]

        y_pred = pd.read_pickle(self.input()["prediction"]["prediction"].path)

        self._compute_metrics(y_test, y_pred)
        self._plot_and_save(y_train, y_test, y_pred)

    def _plot_and_save(self, y_train, y_test, predictions):
        plt.figure(figsize=(16, 8))

        sns.lineplot(data=y_train[TARGET_NAME], marker="o", color="black", label="Train")
        sns.lineplot(data=y_test[TARGET_NAME], marker="o", color="orange", label="Test")

        pred_col_name = list(predictions.columns)[0]

        sns.lineplot(data=predictions[pred_col_name], marker="o", color="green", label=pred_col_name)

        plt.legend()
        plt.title(f"Prediction Visualization\n\nRMSE: {self.rmse:.5f}\nMAE: {self.mae:.5f}", loc="left")

        plt.tight_layout()
        plt.savefig(self.output().path)
        plt.close()

    def _compute_metrics(self, y_test, predictions):
        self.rmse = mean_squared_error(y_test, predictions, squared=False)
        self.mae = mean_absolute_error(y_test, predictions)


class ScoreAndVisualizeUnivariateTSPrediction(ScoreAndVisualizePredictions):
    abstract = False

    split_dataset = ClsParameter(tpe=SimpleSplitTS.return_type())
    univariate_ts_prediction = ClsParameter(tpe=UnivariateTSPredictor.return_type())

    def requires(self):
        return {
            "split_dataset": self.split_dataset(),
            "prediction": self.univariate_ts_prediction()
        }


class ScoreAndVisualizeTabularTSPrediction(ScoreAndVisualizePredictions):
    abstract = False

    split_dataset = ClsParameter(tpe=SimpleSplitTS.return_type())
    tabular_ts_prediction = ClsParameter(tpe=TabularTSPredictor.return_type())

    def requires(self):
        return {
            "split_dataset": self.split_dataset(),
            "prediction": self.tabular_ts_prediction()
        }


if __name__ == '__main__':

    target = ScoreAndVisualizePredictions.return_type()
    print("Collecting Repo")
    rm = RepoMeta
    repository = rm.repository
    # StaticJSONRepo(rm).dump_static_repo_json()

    print("Build Repository...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(rm.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_pipelines_when_infinite = 2000
    actual = inhabitation_result.size()
    max_results = max_pipelines_when_infinite
    if actual > 0:
        max_results = actual

    print("Validating results...")
    unfiltered_results = [t() for t in inhabitation_result.evaluated[0:max_results]]
    print("Number of unfiltered results", len(unfiltered_results))

    validator = SubPipelineSequenceValidator(
        sub_pipeline_template=[AddLagColumn, AddMonthYearColumn,
                               AddExponentialSmoothing2ndOrderColumn,
                               AddExponentialSmoothing3rdOrderColumn])


    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]
    print("Number of results", len(results))

    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()

        print("Run Pipelines")
        luigi.build(results,
                    local_scheduler=False)
    else:
        print("No results!")

    for r in results:
        save_deps_tree_with_task_id(r)

    print("Number of results", len(results))
