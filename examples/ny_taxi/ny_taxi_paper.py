from os.path import exists, join
import pickle
from pathlib import Path
import luigi
import pandas as pd
import json
from pickle import dump

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from cls_luigi.inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi_read_tabular_data import WriteSetupJson, ReadTabularData

import seaborn as sns
import matplotlib.pyplot as plt

from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

sns.set_style('darkgrid')
sns.set_context('talk')

PATH = "data/"


def read_setup(path=join(PATH, 'setup.json')):
    with open(path, 'rb') as f:
        setup = json.load(f)
    return setup


class WriteCSVRegressionSetupJson(WriteSetupJson):
    abstract = False

    def run(self):
        d = {
            "csv_file": "data/taxy_trips_ny_2016-06-01to03_3%sample.csv",
            "date_column": ['pickup_datetime'],
            # "temperature_column": ["max_temp", "min_temp"],
            "drop_column": ["s_fall", "has_snow", "has_rain", "s_depth",
                            "main_street", "trip_distance",
                            'pickup_datetime', 'dropoff_datetime',
                            'vendor_id', "passenger_count", "max_temp", "min_temp",
                            "pickup_zipcode", "dropoff_zipcode", "totalturn",
                            "totalleft", "haversine_distance", "totalsteps"],
            "target_column": 'trip_duration',
            "seed": 42,
            "reg_params": {
                "decision_tree": {
                    "max_depth": 23,
                    "min_samples_leaf": 100,  # min_child_weight
                    "random_state": 42
                },
                "random_forest": {
                    "max_depth": 28,
                    "max_features": 0.8,  # colsample_bytree
                    "max_samples": 0.9,  # subsample
                    "n_estimators": 80,  # n_trees
                    "random_state": 42
                },
                "extra_trees": {
                    "n_estimators": 60,  # n_trees
                    "max_depth": 28,  # n_trees
                    "max_features": 0.6,  # colsample_bytree
                    "random_state": 42
                },
                "xgb": {
                    "eta": 0.01,
                    "lambda": 1,
                    "max_depth": 21,
                    "gamma": 0,
                    "min_child_weight": 110,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "seed": 42

                },
                "light_gbm": {
                    "max_leaves": 900,
                    "min_data_in_leaf": 130,
                    "gamma": 20,
                    "lambda": 1,
                    "learning_rate": 0.05,
                    "seed": 42

                }

            }
        }
        with open(self.output().path, 'w') as f:
            json.dump(d, f, indent=4)


class ReadTaxiData(ReadTabularData):
    abstract = False

    def run(self):
        setup = self._read_setup()
        taxi = pd.read_csv(setup["csv_file"], parse_dates=setup["date_column"])
        taxi.to_pickle(self.output().path)


class DropDuplicatesAndNA(luigi.Task, LuigiCombinator):
    abstract = False
    taxi_data = ClsParameter(tpe=ReadTaxiData.return_type())

    def requires(self):
        return [self.taxi_data()]

    def run(self):
        taxi = pd.read_pickle(self.input()[0].path)
        print("Taxi DataFrame shape before dropping NaNs and duplicates", taxi.shape)
        taxi = taxi.dropna().drop_duplicates()
        print("Taxi DataFrame shape after dropping NaNs and duplicates", taxi.shape)
        taxi.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget(join(PATH, 'cleaned_data.pkl'))


class FilterImplausibleTrips(luigi.Task, LuigiCombinator):
    abstract = False
    clean_data = ClsParameter(tpe=DropDuplicatesAndNA.return_type())

    def requires(self):
        return [self.clean_data()]

    def run(self):
        setup = read_setup()
        taxi = pd.read_pickle(self.input()[0].path)
        taxi = taxi[
            (taxi[setup["target_column"]] >= 10) &
            (taxi[setup["target_column"]] <= 100000)
            ]
        print("Taxi DataFrame shape after filtering implausible trips", taxi.shape)
        taxi.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget(join(PATH, 'filtered.pkl'))


class ExtractRawTemporalFeatures(luigi.Task, LuigiCombinator):
    abstract = False
    filtered_trips = ClsParameter(tpe=FilterImplausibleTrips.return_type())

    def requires(self):
        return [self.filtered_trips()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].path)

    def run(self):
        setup = read_setup()
        tabular = self._read_tabular_data()
        raw_temporal_features = pd.DataFrame(index=tabular.index)
        for c in setup["date_column"]:
            print("Preprocessing Datetime-Column:", c)
            raw_temporal_features[c + "_WEEKDAY"] = tabular[c].dt.dayofweek
            raw_temporal_features[c + "_HOUR"] = tabular[c].dt.hour

        raw_temporal_features.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget(join(PATH, 'raw_temporal_features.pkl'))


class AssembleFinalDataset(luigi.Task, LuigiCombinator):
    abstract = False
    filtered_trips = ClsParameter(tpe=FilterImplausibleTrips.return_type())
    raw_temporal_features = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())

    def requires(self):
        return [self.filtered_trips(), self.raw_temporal_features()]

    def _get_variant_label(self):
        var_label_name = list(map(
            lambda local_target: Path(local_target.path).stem, self.input()))
        return "-".join(var_label_name)

    def run(self):
        setup = read_setup()
        df_joined = None
        for i in self.input():
            path = i.path
            if df_joined is None:
                df_joined = pd.read_pickle(path)
            else:
                data = pd.read_pickle(path)
                df_joined = pd.merge(df_joined, data, left_index=True, right_index=True)

        if len(setup["drop_column"]) != 0:
            df_joined = df_joined.drop(setup["drop_column"], axis="columns")
        df_joined.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget(join(PATH, "final_dataset-" + self._get_variant_label() + ".pkl"))


class TrainTestSplit(luigi.Task, LuigiCombinator):
    abstract = False
    final_dataset = ClsParameter(tpe=AssembleFinalDataset.return_type())
    test_size = luigi.FloatParameter(default=0.33)

    def requires(self):
        return [self.final_dataset()]

    def _get_variant_label(self):
        return Path(self.input()[0].path).stem

    def _read_tabular(self):
        return pd.read_pickle(self.input()[0].path)

    def run(self):
        tabular = self._read_tabular()
        setup = read_setup()

        train, test = train_test_split(tabular, test_size=self.test_size, random_state=setup['seed'])

        train.to_pickle(self.output()["train"].path)
        test.to_pickle(self.output()["test"].path)

    def output(self):
        return {"train": luigi.LocalTarget(join(PATH, 'train-' + self._get_variant_label() + '.pkl')),
                "test": luigi.LocalTarget(join(PATH, 'test-' + self._get_variant_label() + '.pkl'))}


class FitTransformScaler(luigi.Task, LuigiCombinator):
    abstract = True
    splitted_data = ClsParameter(tpe=TrainTestSplit.return_type())
    scaled_files_label = luigi.Parameter()  # string to be used in the output naming
    scaler_label = luigi.Parameter()  # string to be used in the output naming

    def requires(self):
        return [self.splitted_data()]

    def _read_training_data(self):
        return pd.read_pickle(self.input()[0]["train"].path)

    def _read_testing_data(self):
        return pd.read_pickle(self.input()[0]["test"].path)

    def _get_training_variant_label(self):
        return Path(self.input()[0]["train"].path).stem

    def _get_testing_variant_label(self):
        return Path(self.input()[0]["test"].path).stem

    def run(self):
        setup = read_setup()
        train = self._read_training_data()
        X = train.drop(setup["target_column"], axis="columns")
        y = train[[setup["target_column"]]]
        scaler = self._init_scaler()
        scaler.fit(X)
        scaled = pd.DataFrame(scaler.transform(X),
                              columns=scaler.feature_names_in_,
                              index=X.index)
        scaled[setup["target_column"]] = y
        scaled.to_pickle(self.output()["scaled_train"].path)

        test = self._read_testing_data()
        X = test.drop(setup["target_column"], axis="columns")
        y = test[[setup["target_column"]]]
        scaled = pd.DataFrame(scaler.transform(X),
                              columns=scaler.feature_names_in_,
                              index=X.index)
        scaled[setup["target_column"]] = y
        scaled.to_pickle(self.output()["scaled_test"].path)

        with open(self.output()["scaler"].path, 'wb') as outfile:
            pickle.dump(scaler, outfile)

    def output(self):
        return {
            "scaled_train": luigi.LocalTarget(
                join(PATH, self._get_training_variant_label() + "-" + self.scaled_files_label + '.pkl')),
            "scaled_test": luigi.LocalTarget(
                join(PATH, self._get_testing_variant_label() + "-" + self.scaled_files_label + '.pkl')),
            "scaler": luigi.LocalTarget(
                join(PATH, self.scaler_label + "-" + self._get_training_variant_label() + '.pkl'))}

    def _init_scaler(self):  # this method may be used to initialize the scaler and to pass some parameters as well
        pass


class FitTransformRobustScaler(FitTransformScaler):
    abstract = False
    scaled_files_label = luigi.Parameter(default="robust_scaled")
    scaler_label = luigi.Parameter(default="robust_scaler")

    def _init_scaler(self):
        return RobustScaler()


class FitTransformMinMaxScaler(FitTransformScaler):
    abstract = False
    scaled_files_label = luigi.Parameter(default="minmax_scaled")
    scaler_label = luigi.Parameter(default="minmax_scaler")

    def _init_scaler(self):
        return MinMaxScaler()


class FitTransformStandardScaler(FitTransformScaler):
    abstract = False
    scaled_files_label = luigi.Parameter(default="standard_scaled")
    scaler_label = luigi.Parameter(default="standard_scaler")

    def _init_scaler(self):
        return StandardScaler()


class TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    scaled_data = ClsParameter(tpe=FitTransformScaler.return_type())
    regressor_label = luigi.Parameter()  # string to be used in the output naming

    def requires(self):
        return [self.scaled_data()]

    def _get_variant_label(self):
        return Path(self.input()[0]["scaled_train"].path).stem

    def _read_scaled_training_data(self):
        return pd.read_pickle(self.input()[0]["scaled_train"].path)

    def run(self):
        setup = read_setup()
        regressor = self._init_regressor()

        print("TARGET:", setup["target_column"])
        print("NOW WE FIT {} MODEL".format(regressor.__class__.__name__))
        train_data = self._read_scaled_training_data()
        X = train_data.drop(setup["target_column"], axis="columns")
        y = train_data[[setup["target_column"]]].values.ravel()
        print("WITH THE FEATURES")
        print(X.columns)
        print(X)
        print(X.shape)
        print("AND Target")
        print(y)
        print(y.shape)

        regressor.fit(X, y)
        with open(self.output()[0].path, 'wb') as f:
            dump(regressor, f)

    def output(self):
        return [
            luigi.LocalTarget(join(PATH, self.regressor_label + "-" + self._get_variant_label() + '.pkl'))]

    def _init_regressor(
            self):  # this method may be used to initialize the regressor and to pass some parameters as well
        pass


class TrainDecisionTreeModel(TrainRegressionModel):
    abstract = False
    regressor_label = luigi.Parameter(default="decision_tree")

    def _init_regressor(self):
        params = read_setup()["reg_params"]["decision_tree"]
        return DecisionTreeRegressor(**params)
        # return DecisionTreeRegressor()


class TrainExtraTreeModel(TrainRegressionModel):
    abstract = False
    regressor_label = luigi.Parameter(default="extra_tree")

    def _init_regressor(self):
        params = read_setup()["reg_params"]["extra_trees"]
        return ExtraTreesRegressor(**params)
        # return ExtraTreeRegressor()


class TrainRandomForrestModel(TrainRegressionModel):
    abstract = False
    regressor_label = luigi.Parameter(default="random_forest")

    def _init_regressor(self):
        params = read_setup()["reg_params"]["random_forest"]
        return RandomForestRegressor(**params)
        # return RandomForestRegressor()


class TrainXgBoostModel(TrainRegressionModel):
    abstract = False
    regressor_label = luigi.Parameter(default="xgboost")

    def _init_regressor(self):
        params = read_setup()["reg_params"]["xgb"]
        return XGBRegressor(**params)


class TrainLightGBMModel(TrainRegressionModel):
    abstract = False
    regressor_label = luigi.Parameter(default="light_gbm")

    def _init_regressor(self):
        params = read_setup()["reg_params"]["light_gbm"]
        return lgb.LGBMRegressor(**params)


class Predict(luigi.Task, LuigiCombinator):
    abstract = False
    regressor = ClsParameter(tpe=TrainRegressionModel.return_type())
    scaled_data = ClsParameter(tpe=FitTransformScaler.return_type())

    def requires(self):
        return [self.regressor(), self.scaled_data()]

    def _load_regressor(self):
        with open(self.input()[0][0].path, 'rb') as file:
            reg = pickle.load(file)

        return reg

    def _read_scaled_test_data(self):
        p = self.input()[1]["scaled_test"].path
        with open(p, 'rb') as file:
            test_data = pickle.load(file)

        return test_data

    def _get_variant_label(self):
        return Path(self.input()[0][0].path).stem

    def run(self):
        reg = self._load_regressor()
        test_data = self._read_scaled_test_data()
        setup = read_setup()
        y_true_and_pred = pd.DataFrame(index=test_data.index)
        X = test_data.drop(setup["target_column"], axis="columns")
        y_true_and_pred[setup["target_column"]] = test_data[setup["target_column"]].values.ravel()
        y_true_and_pred['prediction'] = reg.predict(X)

        with open(self.output().path, 'wb') as outfile:
            pickle.dump(y_true_and_pred, outfile)

    def output(self):
        return luigi.LocalTarget(join(PATH, 'true_and_prediction' + "-" + self._get_variant_label() + '.pkl'))


class EvaluateAndVisualize(luigi.Task, LuigiCombinator):
    abstract = False
    y_true_and_pred = ClsParameter(tpe=Predict.return_type())
    sort_by = luigi.ChoiceParameter(default="RMSE", choices=["RMSE", "MAE", "R2"])

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
        return self.y_true_and_pred()

    def _read_y_true_and_prediction(self):
        setup = read_setup()
        tabular = pd.read_pickle(self.input().path)
        return tabular[setup['target_column']], tabular['prediction']

    def _get_reg_name(self):
        p = Path(self.input().path).name
        p = p.split('-')[1:]
        return '-'.join(p)

    def _compute_metrics(self, y_true, y_pred):
        self.rmse = round(mean_squared_error(y_true, y_pred, squared=False), 3)
        self.mae = round(mean_absolute_error(y_true, y_pred), 3)
        self.r2 = round(r2_score(y_true, y_pred), 3)

    def _update_leaderboard(self, show_summary=False):
        if exists(self.output()[0].path) is False:
            self.leaderboard = pd.DataFrame(columns=["regressor", "RMSE", "MAE", "R2"])
        else:
            self.leaderboard = pd.read_csv(self.output()[0].path, index_col="index")

        if self._get_reg_name() not in self.leaderboard["regressor"].values:
            self.leaderboard.loc[self.leaderboard.shape[0]] = [self._get_reg_name(), self.rmse, self.mae, self.r2]
            self.leaderboard = self.leaderboard.sort_values(by=self.sort_by, ascending=True)
            self.leaderboard.to_csv(self.output()[0].path, index_label="index")
            self._visualize_leaderboard(show=show_summary)
        else:
            print("scores already exist for: ", self._get_reg_name())

    def _visualize_leaderboard(self, top_results_max_limit=5, show=False):
        top_models = self.leaderboard.head(top_results_max_limit)

        top_models = top_models.reset_index()
        top_models = top_models.rename(columns={
            "index": "Leaderboard Index",
            "RMSE": "Root Mean Squared Error",
            "MAE": "Mean Absolute Error",
            "R2": "Coefficient of Determination (R\u00b2)"
        })

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle(
            "\nLeaderboard Top {} Models Metrics\nSorted by:{}\n\n".format(len(top_models), self.sort_by),
            x=0.05, ha="left")

        order = top_models["Leaderboard Index"].values

        sns.barplot(x="Leaderboard Index", y="Root Mean Squared Error", data=top_models, ax=axes[0], order=order)
        sns.barplot(x="Leaderboard Index", y="Mean Absolute Error", data=top_models, ax=axes[1], order=order)
        sns.barplot(x="Leaderboard Index", y="Coefficient of Determination (R\u00b2)", data=top_models, ax=axes[2],
                    order=order)

        plt.tight_layout()
        plt.savefig(self.output()[1].path)
        if show:
            plt.show()
        plt.close(fig)

    def _visualize_model(self, y_true, y_pred, show=False):

        if exists(self.output()[2].path) is False:

            fig, axes = plt.subplots(2, 2, figsize=(17, 12))
            fig.suptitle(
                "\nPerformance Evaluation\nRMSE: {}\nMAE: {}\nR\u00b2: {}\n".format(self.rmse, self.mae, self.r2),
                x=0.05, ha="left")

            df_prediction = pd.DataFrame({
                'Actual': y_true,
                'Predicted': y_pred})

            def compute_residual(row):
                return row['Predicted'] - row['Actual']

            df_prediction['Prediction Residual'] = df_prediction.apply(
                compute_residual, axis=1)

            residual_plot = sns.scatterplot(x='Actual', y='Prediction Residual',
                                            data=df_prediction, color='r', ax=axes[0][0])
            residual_plot.set_title('Residual Scatter Plot\nIdeal Situation (Predicted = Actual) in Green', loc='left')
            residual_plot.axhline(
                y=0, color='green',
                ls='-', lw=3)

            scatter_plot = sns.scatterplot(x='Actual', y='Predicted',
                                           data=df_prediction, color='r', ax=axes[0][1])
            scatter_plot.set_title(
                'Predictions -- Ground Truth Scatter plot\n'
                'Ideal Situation (Predicted = Actual) in Green', loc='left')
            xlims = (-10, max(max(y_true), max(y_pred)))
            scatter_plot.plot(xlims, xlims, color='g', ls="-", lw=3)

            histogram_residuals = sns.histplot(data=df_prediction, x='Prediction Residual',
                                               kde=True, stat="density", ax=axes[1][0])
            histogram_residuals.set_title('Residuals Histogram\nIdeal Situation (Predicted = Actual) in Green ',
                                          loc='left')
            histogram_residuals.axvline(
                x=0, color='green',
                ls='-', lw=3)

            df_prediction["i"] = df_prediction.index
            df_histplot = pd.melt(df_prediction, id_vars=['i'],
                                  value_vars=['Actual', 'Predicted'],
                                  var_name='Curve', value_name='Target Variable')

            histogram_target_value = sns.histplot(data=df_histplot, x='Target Variable', hue="Curve",
                                                  kde=True, stat="density", ax=axes[1][1])
            histogram_target_value.set_title('Ground Truth & Prediction Values Histogram', loc='left')

            plt.tight_layout()
            plt.savefig(self.output()[2].path)
            if show:
                plt.show()
            plt.close(fig)
        else:
            print(self.output()[2].path, ' already exists!')

    def run(self):
        y_true, y_pred = self._read_y_true_and_prediction()
        self._compute_metrics(y_true, y_pred)
        self._update_leaderboard(show_summary=False)
        self._visualize_model(y_true, y_pred, show=False)
        self.done = True

    def output(self):
        return [luigi.LocalTarget(join(PATH, 'leaderboard.csv')),
                luigi.LocalTarget(join(PATH, "leaderboard_summary.png")),
                luigi.LocalTarget(join(PATH, self._get_reg_name() + ".png"))]


if __name__ == '__main__':
    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = EvaluateAndVisualize.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Build Repository...")
    StaticJSONRepo(RepoMeta).dump_static_repo_json()
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    validator = UniqueTaskPipelineValidator([FitTransformScaler, TrainRegressionModel])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]


    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")
