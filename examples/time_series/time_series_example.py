import os
import pickle

import pandas as pd
from luigi import LocalTarget, Task, build
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt

from cls_luigi.inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from os.path import join as pjoin
from os import mkdir
import pathlib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

#Ideas:

# imputation
# categorical encoding
# Expo Smoothing
# Sliding Window
# Feature Generation
# Feat Selection
# Scaling
# To Supervised learning --> Random Forest, Linear reg, SVR
# easy forecasting methods

#todo exp. smoothing (2 knoten machen)
#todo vereinfachen!!


DIR = os.path.dirname(__file__)
OUTPUTS_DIR = pjoin(DIR, "data")
INPUT_DATA_NAME = "AirPassengers.csv"
INPUT_DATA_PATH = pjoin(DIR, INPUT_DATA_NAME)
TARGET_NAME = "Actual"
TRAIN_TO_INDEX = 120
TEST_PERIODS = 24
DATE_COLS_LIST = ["Date"]


class BaseClass(Task, LuigiCombinator):
    abstract = False

    def _get_last_output_name(self, key):
        name = self.input()[key].path
        name = pathlib.Path(name)
        name = name.stem
        return name


class MakeOutputDirIfNotExist(BaseClass):
    abstract = False

    def run(self):
        print("============= MakeOutputDirIfNotExist RUN")

        mkdir(OUTPUTS_DIR)

    def output(self):
        print("============= MakeOutputDirIfNotExist OUTPUT")

        return LocalTarget(OUTPUTS_DIR)


class ReadTimeSeriesCSV(BaseClass):
    abstract = False
    out_dir = ClsParameter(tpe=MakeOutputDirIfNotExist.return_type())

    def requires(self):
        return self.out_dir()

    def run(self):
        print("============= ReadTimeSeriesCSV RUN")

        time_series = pd.read_csv(INPUT_DATA_PATH, parse_dates=DATE_COLS_LIST)
        time_series.to_csv(self.output().path, index=False, index_label="index")

    def output(self):
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, "time_series.csv"
            )
        )


class SplitData(BaseClass):
    abstract = False
    time_series_data = ClsParameter(tpe=ReadTimeSeriesCSV.return_type())

    def output(self):
        print("============= SplitData OUTPUT")

        return {
            "train": LocalTarget(pjoin(OUTPUTS_DIR, "train.csv")),
            "test": LocalTarget(pjoin(OUTPUTS_DIR, "test.csv")),
        }

    def requires(self):
        return self.time_series_data()

    def run(self):
        print("============= SplitData RUN")

        time_series = pd.read_csv(self.input().path)
        train = time_series.iloc[:TRAIN_TO_INDEX]
        test = time_series.iloc[TRAIN_TO_INDEX:]

        train.to_csv(self.output()["train"].path, index_label="index")
        test.to_csv(self.output()["test"].path, index_label="index")


class GenerateMonthAndYearFeatures(BaseClass):
    abstract = False
    split_data = ClsParameter(tpe=SplitData.return_type())

    def output(self):
        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('train')}-datetime_feats.csv"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('test')}-datetime_feats.csv"
                )
            )
        }

    def requires(self):
        return self.split_data()

    def run(self):
        train = pd.read_csv(self.input()["train"].path, parse_dates=DATE_COLS_LIST, index_col="index")
        test = pd.read_csv(self.input()["test"].path, parse_dates=DATE_COLS_LIST, index_col="index")

        for c in DATE_COLS_LIST:
            train[f"MONTH_{c}"] = train[c].map(self.return_month)
            test[f"MONTH_{c}"] = test[c].map(self.return_month)

            train[f"YEAR_{c}"] = train[c].map(self.return_year)
            test[f"YEAR_{c}"] = test[c].map(self.return_year)

        train = train.drop(DATE_COLS_LIST, axis=1)
        test = test.drop(DATE_COLS_LIST, axis=1)

        train.to_csv(self.output()["train"].path)
        test.to_csv(self.output()["test"].path)

    @staticmethod
    def return_month(dt_obj):
        return dt_obj.month

    @staticmethod
    def return_year(dt_obj):
        return dt_obj.year


class ConvertTargetValuesToFeatures(BaseClass):
    """
    Converts time series data into supervised learning data by converting past target values into features.
    """
    abstract = False
    split_data_with_dt_feats = ClsParameter(tpe=GenerateMonthAndYearFeatures.return_type())
    n_in = 6

    def output(self):
        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('train')}-ts_to_supervised.csv"

                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('test')}-ts_to_supervised.csv"
                )
            )
        }

    def requires(self):
        return self.split_data_with_dt_feats()

    def run(self):
        print("============= ConvertTargetValuesToFeatures RUN")

        train = pd.read_csv(self.input()["train"].path, index_col="index")
        test = pd.read_csv(self.input()["test"].path, index_col="index")

        concatenated = pd.concat([train, test])

        supervised_data = pd.DataFrame(
            columns=[f"t-{i}" for i in range(self.n_in, 0, -1)],
            data=self._get_target_based_features(concatenated),
            index=concatenated.index[self.n_in:]
        )

        supervised_data = supervised_data.join(concatenated)

        supervised_train = supervised_data.iloc[: TRAIN_TO_INDEX - self.n_in]
        supervised_test = supervised_data.iloc[TRAIN_TO_INDEX - self.n_in:]

        supervised_train.to_csv(self.output()["train"].path, index_label="index")
        supervised_test.to_csv(self.output()["test"].path, index="index")

    def _get_target_based_features(self, df):
        cols = []

        # converting targets to features
        for i in range(self.n_in, 0, -1):
            cols.append(df[TARGET_NAME].shift(i))

        supervised_data = pd.concat(cols, axis=1)
        supervised_data.dropna(inplace=True)

        return supervised_data.values


class ExponentialSmoother(BaseClass):
    abstract = True
    supervised_data = ClsParameter(tpe=ConvertTargetValuesToFeatures.return_type())

    smoothing_level = 0.15
    smoothing_trend = 0.05
    smoothing_seasonal = 0.7
    trend_component = "multiplicative"
    season_component = "additive"
    initialization_method = None
    seasonal_periods = 12
    optimized = True

    def requires(self):
        return self.supervised_data()


class ExponentialSmoothing1stOrder(ExponentialSmoother):
    abstract = False

    def output(self):
        print("============= ExponentialSmoothingFirstOrder OUTPUT")

        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('train')}-exp_smt_1st.csv"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('test')}-exp_smt_1st.csv"
                ),
            )
        }

    def run(self):
        print("============= ExponentialSmoothingFirstOrder RUN")

        train = pd.read_csv(self.input()["train"].path, index_col="index")
        test = pd.read_csv(self.input()["test"].path, index_col="index")

        joined = pd.concat([train, test])

        es = SimpleExpSmoothing(
            joined[TARGET_NAME],
            initialization_method=self.initialization_method).fit(
            smoothing_level=self.smoothing_level,
            optimized=self.optimized)

        joined["Exp. Smoothing 1st Order Prediction"] = es.fittedvalues

        train = joined.loc[train.index]
        test = joined.loc[test.index]

        train.to_csv(self.output()["train"].path)
        test.to_csv(self.output()["test"].path)


class ExponentialSmoothing2ndOrder(ExponentialSmoother):
    abstract = False

    def output(self):
        print("============= ExponentialSmoothingSecondOrder OUTPUT")

        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('train')}-exp_smt_2nd.csv"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('test')}-exp_smt_2nd.csv"
                ),
            )
        }

    def run(self):
        print("============= ExponentialSmoothingSecondOrder RUN")

        train = pd.read_csv(self.input()["train"].path, index_col="index")
        test = pd.read_csv(self.input()["test"].path, index_col="index")

        es = Holt(train[TARGET_NAME],
                  initialization_method=self.initialization_method
                  ).fit(
            smoothing_level=self.smoothing_level,
            smoothing_trend=self.smoothing_trend,
            optimized=self.optimized,
        )

        n_forcast_steps = test.shape[0]

        train["Exp. Smoothing 2nd Order Prediction"] = es.fittedvalues
        test["Exp. Smoothing 2nd Order Prediction"] = es.forecast(steps=n_forcast_steps).to_list()

        train.to_csv(self.output()["train"].path)
        test.to_csv(self.output()["test"].path)


class ExponentialSmoothing3rdOrder(ExponentialSmoother):
    abstract = False

    def output(self):
        print("============= ExponentialSmoothingThirdOrder OUTPUT")

        return {
            "train": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('train')}-exp_smt_3rd.csv"
                )
            ),
            "test": LocalTarget(
                pjoin(
                    OUTPUTS_DIR, f"{self._get_last_output_name('test')}-exp_smt_3rd.csv"
                ),
            )
        }

    def run(self):
        print("============= ExponentialSmoothingThirdOrder RUN")

        train = pd.read_csv(self.input()["train"].path, index_col="index")
        test = pd.read_csv(self.input()["test"].path, index_col="index")

        es = ExponentialSmoothing(
            train[TARGET_NAME],
            initialization_method=self.initialization_method,
            trend=self.trend_component,
            seasonal=self.season_component,
            seasonal_periods=self.seasonal_periods,
        ).fit(
            smoothing_level=self.smoothing_level,
            smoothing_trend=self.smoothing_trend,
            smoothing_seasonal=self.smoothing_seasonal,
            optimized=self.optimized,
        )

        n_forcast_steps = test.shape[0]

        train["Exp. Smoothing 3rd Order Prediction"] = es.fittedvalues
        test["Exp. Smoothing 3rd Order Prediction"] = es.forecast(steps=n_forcast_steps).to_list()

        train.to_csv(self.output()["train"].path)
        test.to_csv(self.output()["test"].path)


from cls_luigi.cls_tasks import  ClsBaseTask

class Regression(BaseClass):
    abstract = True
    data = ClsParameter(tpe={
        "with_exp_smoothing": ExponentialSmoother.return_type(),
        "without_exp_smoothing": ConvertTargetValuesToFeatures.return_type()
    })

    config_domain = {"with_exp_smoothing", "without_exp_smoothing"}

    def requires(self):
        return self.data()

    def _get_variant_label(self):
        if self.config_index == "with_exp_smoothing":
            ci_name = self._get_last_output_name("train")
            ci_name = ci_name.split("_")[-1]
            return self.config_index + "_" + ci_name

        else:
            return self.config_index

    def _get_split_train_test_data(self):
        train = pd.read_csv(self.input()["train"].path, index_col="index")
        test = pd.read_csv(self.input()["test"].path, index_col="index")

        x_train = train.drop(TARGET_NAME, axis=1)
        y_train = train[TARGET_NAME]
        x_test = test.drop(TARGET_NAME, axis=1)
        y_test = test[TARGET_NAME]

        return x_train, x_test, y_train, y_test


class FitPredictLinearRegression(Regression):
    abstract = False

    def output(self):
        variant_label = self._get_variant_label()

        return {
            "model": LocalTarget(
                pjoin(OUTPUTS_DIR, f"model_linear_reg-{variant_label}.pkl")
            ),
            "prediction": LocalTarget(
                pjoin(OUTPUTS_DIR, f"prediction_linear_reg-{variant_label}.csv")
            )
        }

    def run(self):
        print(self.config_index)
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()

        reg = LinearRegression()
        reg.fit(x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)

        pred = pd.DataFrame(
            data=reg.predict(x_test),
            index=x_test.index,
            columns=[f"Linear Regression Prediction ({self._get_variant_label()})"]
        )
        pred.to_csv(self.output()["prediction"].path)


class FitPredictRandomForest(Regression):
    abstract = False

    def output(self):
        variant_label = self._get_variant_label()

        return {
            "model": LocalTarget(
                pjoin(OUTPUTS_DIR, f"model_random_forest-{variant_label}.pkl")
            ),
            "prediction": LocalTarget(
                pjoin(OUTPUTS_DIR, f"prediction_random_forest-{variant_label}.csv")
            )
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()

        reg = RandomForestRegressor()
        reg.fit(x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)

        pred = pd.DataFrame(
            data=reg.predict(x_test),
            index=x_test.index,
            columns=[f"Random Forest Prediction ({self._get_variant_label()})"]

        )

        pred.to_csv(self.output()["prediction"].path)


class FitPredictKNeighbors(Regression):
    abstract = False

    def output(self):
        variant_label = self._get_variant_label()

        return {
            "model": LocalTarget(
                pjoin(OUTPUTS_DIR, f"model_knn-{variant_label}.pkl")
            ),
            "prediction": LocalTarget(
                pjoin(OUTPUTS_DIR, f"prediction_knn-{variant_label}.csv")
            )
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()

        reg = KNeighborsRegressor()
        reg.fit(x_train, y_train)

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(reg, outfile)

        pred = pd.DataFrame(
            data=reg.predict(x_test),
            index=x_test.index,
            columns=[f"KNN Prediction ({self._get_variant_label()})"]
        )

        pred.to_csv(self.output()["prediction"].path)


class ScoreAndVisualizePredictions(BaseClass):
    abstract = False

    dataset = ClsParameter(tpe=ConvertTargetValuesToFeatures.return_type())
    predictions = ClsParameter(tpe={
        "ml_model": Regression.return_type(),
        "exp_smoothing": ExponentialSmoother.return_type()
    })

    config_domain = {"ml_model", "exp_smoothing"}

    rmse = None
    mae = None
    r2 = None

    def output(self):
        variant_label = self._get_variant_label()
        return LocalTarget(
            pjoin(
                OUTPUTS_DIR, f"visual_{variant_label}.png"
            )
        )

    def requires(self):
        return {
            "dataset": self.dataset(),
            "predictions": self.predictions()
        }

    def run(self):
        train_data = pd.read_csv(self.input()["dataset"]["train"].path, index_col="index")
        test_data = pd.read_csv(self.input()["dataset"]["test"].path, index_col="index")

        if self.config_index == "ml_model":
            predictions = pd.read_csv(self.input()["predictions"]["prediction"].path, index_col="index")
        elif self.config_index == "exp_smoothing":
            predictions = pd.read_csv(self.input()["predictions"]["test"].path, index_col="index")
            pred_col_name = list(predictions.columns)
            pred_col_name = list(filter(lambda x: "Exp. Smoothing" in x, pred_col_name))
            predictions = predictions[pred_col_name]

        self._compute_scores(test_data, predictions)
        self._make_and_save_plot(train_data, test_data, predictions)

    def _make_and_save_plot(self, train_data, test_data, prediction):
        plt.figure(figsize=(16, 8))

        sns.lineplot(data=train_data, y=TARGET_NAME, x=train_data.index, marker="o", color="black", label="TRAIN")
        sns.lineplot(data=test_data, y=TARGET_NAME, x=test_data.index, marker="o", color="orange", label="TEST")

        sns.lineplot(data=prediction, y=prediction.columns[0], x=prediction.index, marker="o", color="green",
                     label=prediction.columns[0])

        plt.title(f"Prediction Visualization\n\nRMSE: {self.rmse:.5f}\nMAE: {self.mae:.5f}", loc="left")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output().path)

    def _compute_scores(self, test_data, predictions):

        self.rmse = mean_squared_error(test_data[TARGET_NAME], predictions, squared=False)
        self.mae = mean_absolute_error(test_data[TARGET_NAME], predictions)

    def _get_variant_label(self):
        if self.config_index == "ml_model":
            label = self.input()["predictions"]["prediction"].path
            label = pathlib.Path(label)
            label = label.stem
            return label

        elif self.config_index == "exp_smoothing":
            label = self.input()["predictions"]["test"].path
            label = pathlib.Path(label)
            label = label.stem
            label = label.split("-")[-1]
            return label


if __name__ == "__main__":
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = ScoreAndVisualizePredictions.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    #StaticJSONRepo(RepoMeta).dump_static_repo_json()

    # print(deep_str(inhabitation_result.rules))
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    validator = UniqueTaskPipelineValidator([Regression, ExponentialSmoother])

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
