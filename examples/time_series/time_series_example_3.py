import json
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from luigi import LocalTarget, build

from cls.debug_util import deep_str
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.cls_tasks import ClsTask
from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo
from cls_luigi.util.util import save_deps_tree_with_task_id

INPUT_DATA_NAME = "AirPassengers.csv"  # time series to read
TARGET_NAME = "Actual"
TRAIN_TO_INDEX = 120  # used for splitting the time series into train and test sets
FORCAST_PERIOD = 24  # number of periods to predict
DATE_INDEX_COL = "Date"  # name of the column containing the date
SEED = 123  # seed for reproducibility


class ReadTimeSeries(ClsTask):
    """
    Reads time series data from a csv file and converts it to a pandas DataFrame.
    The DataFrame is then pickled

    :input: None
    :output: LocalTarget that points at a pickled pandas.DataFrame named time_series
    """
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
            freq=time_series.index.inferred_freq)

        time_series.to_pickle(self.output().path)


class SplitTimeSeriesData(ClsTask):
    """
    Splits the time series data into train and test sets according to the predefined TRAIN_TO_INDEX variable.
    The train and test sets are then pickled.

    :input: output of ReadTimeSeries
    :output: Dict:
             key: train, value: LocalTargets that point at a pickled pandas.DataFrame
             key: test, value: LocalTargets that point at a pickled pandas.DataFrame
    """

    abstract = False
    time_series_data = ClsParameter(tpe=ReadTimeSeries.return_type())

    def requires(self):
        return self.time_series_data()

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl')),
        }

    def run(self):
        time_series = pd.read_pickle(self.input().path)
        train = time_series.iloc[:TRAIN_TO_INDEX]
        test = time_series.iloc[TRAIN_TO_INDEX:]

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)


class FitPredictExponentialSmoothingModel(ClsTask):
    """
    Abstract class for fitting an exponential smoothing model to the train data and then predicting
    the test data. The fitted values, the predictions, and the model are then pickled.
    Only one concrete implementation (child class) will be present in a given pipeline.

    :input: output of SplitTimeSeriesData
    :output: Dict:
                key: fitted_values, value: LocalTarget that points at a pickled pandas.DataFrame
                key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled
                                   statsmodels.tsa.holtwinters.ExponentialSmoothing model
    """
    abstract = True
    split_data = ClsParameter(tpe=SplitTimeSeriesData.return_type())

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


class ExponentialSmoothing1stOrder(FitPredictExponentialSmoothingModel):
    """
    Fits a 1st order exponential smoothing model to the train data and then predicts the test data.
    The fitted values, the predictions, and the model are then pickled.

    :input: output of SplitTimeSeriesData (implemented in the parent class)
    :output: Dict: (implemented in the parent class)
                key: fitted_values, value: LocalTarget that points at a pickled pandas.DataFrame
                key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled
                                   statsmodels.tsa.holtwinters.ExponentialSmoothing model
    """
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


class ExponentialSmoothing2ndOrder(FitPredictExponentialSmoothingModel):
    """
    Fits a 2nd order exponential smoothing model with an additive trend component to the train data and
    then predicts the test data. The fitted values, the predictions, and the model are then pickled.

    :input: output of SplitTimeSeriesData (implemented in the parent class)
    :output: Dict: (implemented in the parent class)
                key: fitted_values, value: LocalTarget that points at a pickled pandas.DataFrame
                key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled
                                   statsmodels.tsa.holtwinters.ExponentialSmoothing model
    """
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


class ExponentialSmoothing3rdOrder(FitPredictExponentialSmoothingModel):
    """
    Fits a 3rd order exponential smoothing model with multiplicative trend and seasonal components to
    the train data and then predicts the test data. The fitted values, the predictions, and the model are then pickled.


    :input: output of SplitTimeSeriesData (implemented in the parent class)
    :output: Dict: (implemented in the parent class)
                key: fitted_values, value: LocalTarget that points at a pickled pandas.DataFrame
                key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled
                                   statsmodels.tsa.holtwinters.ExponentialSmoothing model
    """
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


class TimeSeriesToSupervisedLearningDataset(ClsTask):
    """
    Transforms a time series into a supervised learning dataset by adding previous values of
    the target variable as features. The number of previous values to add is specified by
    the n_in parameter. The resulting train and test datasets is then pickled.


    :input: output SplitTimeSeriesData
    :output: Dict:
                key: train, value: LocalTarget that points at a pickled pandas.DataFrame
                key: test, value: LocalTarget that points at a pickled pandas.DataFrame
    """
    abstract = False
    split_data = ClsParameter(tpe=SplitTimeSeriesData.return_type())

    n_in = 6

    def requires(self):
        return self.split_data()

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl'))
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


class FeaturePreprocessors(ClsTask):
    """
    Abstract class for feature preprocessors. Only one concrete implementation (child class)
    will be present in a given pipeline.


    :input: output of TimeSeriesToSupervisedLearningDataset
    :output: implemented and specified in the concreate implementation class
    """
    abstract = True
    supervised_learning_data = ClsParameter(tpe=TimeSeriesToSupervisedLearningDataset.return_type())

    def requires(self):
        return self.supervised_learning_data()


class GenerateMonthAndYearColumns(FeaturePreprocessors):
    """
     Adds the month and year as integer columns to the supervised learning dataset from DateTime Column "Date".
     The resulting train and test datasets are then pickled.

    :input: output of  TimeSeriesToSupervisedLearningDataset (implemented in the parent class)
    :output: Dict:
                key: train, value: LocalTarget that points at a pickled pandas.DataFrame
                key: test, value: LocalTarget that points at a pickled pandas.DataFrame
    """
    abstract = False

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl'))
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


class ExponentialSmoothing2ndOrderPreprocessor(FeaturePreprocessors):
    """
    Trains a 2nd order exponential smoothing model and adds the fitted values of to the training data and
    a forcast of length FORCAST_PERIOD to the test data. The resulting train and test datasets are then pickled.


    :input: output of TimeSeriesToSupervisedLearningDataset (implemented in the parent class)
    :output: Dict:
                key: train, value: LocalTarget that points at a pickled pandas.DataFrame
                key: test, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled statsmodels.tsa.holtwinters.ExponentialSmoothing
    """
    abstract = False

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl')),
            "model": LocalTarget(self.get_unique_output_path("model.pkl"))
        }

    def run(self):
        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)

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


class ExponentialSmoothing3rdOrderPreprocessor(FeaturePreprocessors):
    """
    Trains a 2nd order exponential smoothing model and adds the fitted values of to the training data and
    a forcast of length FORCAST_PERIOD to the test data. The resulting train and test datasets are then pickled.

    :input: output of TimeSeriesToSupervisedLearningDataset (implemented in the parent class)
    :output: Dict:
                key: train, value: LocalTarget that points at a pickled pandas.DataFrame
                key: test, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled statsmodels.tsa.holtwinters.ExponentialSmoothing
    """
    abstract = False

    def output(self):
        return {
            "train": LocalTarget(self.get_unique_output_path('train.pkl')),
            "test": LocalTarget(self.get_unique_output_path('test.pkl')),
            "model": LocalTarget(self.get_unique_output_path("model.pkl"))
        }

    def run(self):
        train = pd.read_pickle(self.input()["train"].path)
        test = pd.read_pickle(self.input()["test"].path)

        es = ExponentialSmoothing(
            train[TARGET_NAME],
            trend="mul",
            seasonal="mul"
        ).fit()

        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(es, outfile)

        train["Exp. Smoothing 3rd Order"] = es.fittedvalues
        test["Exp. Smoothing 3rd Order"] = es.forecast(FORCAST_PERIOD)

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)


class NoFeaturePreprocessing(FeaturePreprocessors):
    """
    Does not add any features to the supervised learning dataset. It just passes the input data as an output.
    No data will be written permanently here.

    :input: output of TimeSeriesToSupervisedLearningDataset (implemented in the parent class)
    :output: output of TimeSeriesToSupervisedLearningDataset
    """

    abstract = False

    def output(self):
        return self.input()


class FitPredictRegressionModel(ClsTask):
    """
    Abstract class for fitting and predicting values with regression models.
    Since a model can have different configs that are passed to it from a config task, the requirements (dependencies)
     are added in the concreate implementations. The trained model and the predicted values are then pickled.

    :input: implemented and specified in the concrete implementations
    :output: Dict:
                key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
                key: model, value: LocalTarget that points at a pickled sklearn.base.BaseEstimator
    """
    abstract = True

    def output(self):
        return {
            "model": LocalTarget(self.get_unique_output_path('model.pkl')),
            "prediction": LocalTarget(self.get_unique_output_path('prediction.pkl'))}

    def _get_split_train_test_data(self):
        train = pd.read_pickle(self.input()["preprocessed_data"]["train"].path)
        test = pd.read_pickle(self.input()["preprocessed_data"]["test"].path)

        x_train = train.drop(TARGET_NAME, axis=1)
        x_test = test.drop(TARGET_NAME, axis=1)
        y_train = train[TARGET_NAME]
        y_test = test[TARGET_NAME]

        return x_train, x_test, y_train, y_test

    def _load_algorithm_config(self):
        with open(self.input()["config"].path, "r") as infile:
            return json.load(infile)


class LassoConfigs(ClsTask):
    """
    Abstract class for creating and saving a JSON file with parameter values of a Lasso regression model.

    :input: None
    :output: specified in the concrete implementations
    """
    abstract = True


class LassoConfig1(LassoConfigs):
    """
    Creates and saves a JSON file with parameter values of a Lasso regression model.

    :input: None
    ;output: LocalTarget that points at a JSON file with parameter values of a Lasso regression model
    """
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path('lasso_config_1.json'))

    def run(self):
        params = {
            "alpha": 0.5,
            "max_iter": 2000,
            "tol": 1e-4,
            "selection": "cyclic",
            "random_state": SEED
        }

        with open(self.output().path, "w+") as outfile:
            json.dump(params, outfile)


class LassoConfig2(LassoConfigs):
    """
    Creates and saves a JSON file with parameter values of a Lasso regression model.

    :input: None
    :output: JSON file with parameter values of a Lasso regression model
    """
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path('lasso_config_2.json'))

    def run(self):
        params = {
            "alpha": 0.8,
            "max_iter": 2500,
            "tol": 1e-5,
            "selection": "random",
            "random_state": SEED
        }

        with open(self.output().path, "w+") as outfile:
            json.dump(params, outfile)


class FitPredictLasso(FitPredictRegressionModel):
    """
    Fits a Lasso regression model and predicts values for the test data.
    The model's parameters are loaded from a JSON file according to the dependency "LassoConfigs".

    :input: Dict:
            key: preprocessed_data, value: output of a concreate implementation of FeaturePreprocessors
            key: config, value: output of LassoConfigs

    :output: Dict:  (implemented in the parent class)
            key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
            key: model, value: LocalTarget that points at a pickled sklearn.base.BaseEstimator
    """
    abstract = False
    config = ClsParameter(tpe=LassoConfigs.return_type())
    preprocessed_supervised_learning_data = ClsParameter(tpe=FeaturePreprocessors.return_type())

    def requires(self):
        return {
            "preprocessed_data": self.preprocessed_supervised_learning_data(),
            "config": self.config()
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()
        config = self._load_algorithm_config()

        lasso_reg = Lasso(**config).fit(x_train, y_train)
        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(lasso_reg, outfile)

        pred_col_name = "Lasso Regression Prediction with " + \
                        self.requires()['preprocessed_data'].__class__.__name__ + \
                        " and " + self.requires()['config'].__class__.__name__

        prediction = pd.DataFrame(
            data=lasso_reg.predict(x_test),
            index=x_test.index,
            columns=[pred_col_name]
        )
        prediction.to_pickle(self.output()["prediction"].path)


class RandomForestConfigs(ClsTask):
    """
    Abstract class for creating and saving a JSON file with parameter values of a random forest regression model.

    :input: None
    :output: specified in the concrete implementations
    """
    abstract = True


class RandomForestConfig1(RandomForestConfigs):
    """
    Creates and saves a JSON file with parameter values of a random forest regression model.

    :input: None
    :output: LocalTarget that points at a JSON file with parameter values of a random forest regression model
    """
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path('random_forest_config_1.json'))

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
    """
    Creates and saves a JSON file with parameter values of a random forest regression model.

    :input: None
    :output: LocalTarget that points at a JSON file with parameter values of a random forest regression model
    """
    abstract = False

    def output(self):
        return LocalTarget(self.get_unique_output_path('random_forest_config_2.json'))

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
    """
    Fits a random forest regression model and predicts values for the test data.
    The model's parameters are loaded from a JSON file according to the "RandomForestConfigs".

    :input: Dict:
            key: preprocessed_data, value: output of a concreate implementation of FeaturePreprocessors
            key: config, value: output of RandomForestConfigs

    :output: Dict: (implemented in the parent class)
            key: prediction, value: LocalTarget that points at a pickled pandas.DataFrame
            key: model, value: LocalTarget that points at a pickled sklearn.base.BaseEstimator
    """
    abstract = False
    config = ClsParameter(tpe=RandomForestConfigs.return_type())
    preprocessed_supervised_learning_data = ClsParameter(tpe=FeaturePreprocessors.return_type())

    def requires(self):
        return {
            "preprocessed_data": self.preprocessed_supervised_learning_data(),
            "config": self.config()
        }

    def run(self):
        x_train, x_test, y_train, y_test = self._get_split_train_test_data()
        config = self._load_algorithm_config()

        random_forest_reg = RandomForestRegressor(**config).fit(x_train, y_train)
        with open(self.output()["model"].path, "wb") as outfile:
            pickle.dump(random_forest_reg, outfile)

        pred_col_name = "Random Forest Prediction with " + \
                        self.requires()['preprocessed_data'].__class__.__name__ + \
                        " and " + self.requires()['config'].__class__.__name__

        prediction = pd.DataFrame(
            data=random_forest_reg.predict(x_test),
            index=x_test.index,
            columns=[pred_col_name]
        )
        prediction.to_pickle(self.output()["prediction"].path)


class ScoreAndVisualizePredictions(ClsTask):
    """
    Scores and visualizes the predictions of a regression or a time series model.
    This task results in a PNG file that shows the predictions and the actual values of the test data,
    as well as the RMSE and MAE of the predictions.


    :input: Dict:
            key: split_dataset, value: output of SplitTimeSeriesData
            key: prediction, value: output of one concreate implementation of
                                    FitPredictRegressionModel or FitPredictExponentialSmoothingModel

    :output: LocalTarget that points at a PNG file
    """
    abstract = False

    split_dataset = ClsParameter(tpe=SplitTimeSeriesData.return_type())
    prediction = ClsParameter(
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
            "split_dataset": self.split_dataset(),
            "prediction": self.prediction()
        }

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

    def _compute_metrics(self, y_test, predictions):
        self.rmse = mean_squared_error(y_test, predictions, squared=False)
        self.mae = mean_absolute_error(y_test, predictions)


if __name__ == "__main__":
    # specify target (end task)
    target = ScoreAndVisualizePredictions.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    # StaticJSONRepo(RepoMeta).dump_static_repo_json()
    # print(deep_str(inhabitation_result.rules))
    
    # check if there are is infinite number of results
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    # remove invalid pipelines (for this case)
    validator = UniqueTaskPipelineValidator(
        [FitPredictExponentialSmoothingModel, FeaturePreprocessors, FitPredictRegressionModel])
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

    # save txt that shows the dependency trees and the corresponding task ids
    for r in results:
        save_deps_tree_with_task_id(task=r)
