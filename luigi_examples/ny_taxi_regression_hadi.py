import luigi
import pandas as pd
import json
from pickle import dump
from sklearn.linear_model import LinearRegression
from inhabitation_task import LuigiCombinator, ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes
from cls_luigi_read_tabular_data import WriteSetupJson, ReadTabularData


class WriteCSVRegressionSetupJson(WriteSetupJson):
    abstract = False

    def run(self):
        d = {
            "csv_file": "data/taxy_trips_ny_2016-06-01to03_3%sample.csv",
            "date_column": ['pickup_datetime', 'dropoff_datetime'],
            "temperature_column": ["max_temp", "min_temp"],
            "drop_column": ["rain", "has_rain", "main_street",
                            "main_street_ratio", "trip_distance",
                            'pickup_datetime', 'dropoff_datetime',
                            'vendor_id', "passenger_count"],
            "target_column": 'trip_duration',
        }
        with open(self.output().path, 'w') as f:
            json.dump(d, f, indent=4)


class ReadTaxiData(ReadTabularData):
    abstract = False

    def run(self):
        setup = self._read_setup()
        taxi = pd.read_csv(setup["csv_file"], parse_dates=setup["date_column"])
        taxi.to_pickle(self.output().path)


class FilterTabularData(luigi.Task, LuigiCombinator):
    abstract = True
    tabular_data = ClsParameter(tpe=ReadTaxiData.return_type())
    setup = ClsParameter(tpe=WriteCSVRegressionSetupJson.return_type())

    def requires(self):
        return [self.tabular_data(), self.setup()]

    def _read_setup(self):
        with open('data/setup.json') as file:
            setup = json.load(file)
        return setup


class FilterImplausibleTrips(FilterTabularData):
    abstract = False

    def run(self):
        setup = self._read_setup()
        taxi = pd.read_pickle(self.input()[0].open().name)
        print("Taxi DataFrame shape before dropping NaNs and duplicates", taxi.shape)
        taxi = taxi.dropna().drop_duplicates()
        print("Taxi DataFrame shape after dropping NaNs and duplicates", taxi.shape)
        taxi = taxi[
            (taxi[setup["target_column"]] >= 10) &
            (taxi[setup["target_column"]] <= 100000)
            ]
        print("Taxi DataFrame shape after filtering implausible trips", taxi.shape)
        taxi.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/filtered_tabular_data.pkl')


class ExtractRawTemporalFeatures(luigi.Task, LuigiCombinator):
    abstract = False
    filtered_tabular_data = ClsParameter(tpe=FilterTabularData.return_type())
    setup = ClsParameter(tpe=WriteCSVRegressionSetupJson.return_type())

    def requires(self):
        return [self.filtered_tabular_data(), self.setup()]

    def _read_setup(self):
        with open('data/setup.json') as file:
            setup = json.load(file)
        return setup

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        setup = self._read_setup()
        tabular = self._read_tabular_data()
        raw_temporal_features = pd.DataFrame(index=tabular.index)
        for c in setup["date_column"]:
            print("Preprocessing Datetime-Column:", c)
            raw_temporal_features[c + "_YEAR"] = tabular[c].dt.year
            raw_temporal_features[c + "_MONTH"] = tabular[c].dt.hour
            raw_temporal_features[c + "_DAY"] = tabular[c].dt.day
            raw_temporal_features[c + "_WEEKDAY"] = tabular[c].dt.dayofweek
            raw_temporal_features[c + "_HOUR"] = tabular[c].dt.hour

        raw_temporal_features.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/raw_temporal_features.pkl')


class DummyEncodePickupInWeekend(luigi.Task, LuigiCombinator):
    abstract = False
    raw_temporal_data = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())

    def requires(self):
        return [self.raw_temporal_data()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        raw_temporal_data = self._read_tabular_data()
        df_is_weekend = pd.DataFrame(index=raw_temporal_data.index)

        def weekend_mapping(weekday):
            if weekday >= 5:
                return 1
            return 0

        df_is_weekend["is_weekend"] = raw_temporal_data["pickup_datetime_WEEKDAY"].map(
            weekend_mapping)
        df_is_weekend.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/is_weekend.pkl')


class DummyEncodePickupBeforeAfter7AM(luigi.Task, LuigiCombinator):
    abstract = False
    raw_temporal_data = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())

    def requires(self):
        return [self.raw_temporal_data()]

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        raw_temporal_data = self._read_tabular_data()
        df_after_7am = pd.DataFrame(index=raw_temporal_data.index)

        def after_7am_mapping(hour):
            if hour >= 7:
                return 1
            return 0

        df_after_7am["is_after_7am"] = raw_temporal_data["pickup_datetime_HOUR"].map(
            after_7am_mapping)
        df_after_7am.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/is_after_7am.pkl')


class PreprocessorDummy(DummyEncodePickupInWeekend, DummyEncodePickupBeforeAfter7AM):
    abstract = False
    tabular_data = ClsParameter(tpe=ReadTabularData.return_type())

    def run(self):
        tabular = self._read_tabular_data()
        tabular.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget('data/tabular_data_preprocessed_dummy.pkl')


s = {DummyEncodePickupInWeekend, DummyEncodePickupBeforeAfter7AM}


class PreprocessorSet(luigi.Task, LuigiCombinator):
    abstract = False
    filtered_trips = ClsParameter(tpe=FilterTabularData.return_type())
    raw_temporal_features = ClsParameter(tpe=ExtractRawTemporalFeatures.return_type())
    is_weekend = ClsParameter(tpe=DummyEncodePickupInWeekend.return_type() \
        if DummyEncodePickupInWeekend in s else PreprocessorDummy.return_type())

    is_after_7am = ClsParameter(tpe=DummyEncodePickupBeforeAfter7AM.return_type() \
        if DummyEncodePickupBeforeAfter7AM in s else PreprocessorDummy.return_type())

    def requires(self):
        return [self.filtered_trips(), self.raw_temporal_features(),
                self.is_weekend(), self.is_after_7am()]

    def output(self):
        return [self.input()[0], self.input()[1], self.input()[2], self.input()[3]]


class JoinPreprocessedData(luigi.Task, LuigiCombinator):
    abstract = False
    s = ClsParameter(tpe=PreprocessorSet.return_type())

    def requires(self):
        return self.s()

    def run(self):
        df_joined = None
        for r in self.input():
            if r.open().name != 'data/tabular_data_preprocessed_dummy.pkl':
                if df_joined is None:
                    df_joined = pd.read_pickle(r.open().name)
                else:
                    data = pd.read_pickle(r.open().name)
                    df_joined = pd.merge(df_joined, data, left_index=True, right_index=True)
        df_joined.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget("data/joined_preprocessed_data.pkl")


class FilterColumns(luigi.Task, LuigiCombinator):
    abstract = False
    Joined_data = ClsParameter(tpe=JoinPreprocessedData.return_type())

    def requires(self):
        return [self.Joined_data()]

    def _read_setup(self):
        with open('data/setup.json') as file:
            setup = json.load(file)
        return setup

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)

    def run(self):
        setup = self._read_setup()
        data = self._read_tabular_data()
        if len(setup["drop_column"]) != 0:
            data = data.drop(setup["drop_column"], axis="columns")
        data.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget("data/joined_preprocessed_data_cols_filtered.pkl")


class FitTransformScaler(luigi.Task, LuigiCombinator):
    abstract = True
    tabular_data_preprocessed_filtered = ClsParameter(tpe=FilterColumns.return_type())

    def requires(self):
        return [self.tabular_data_preprocessed_filtered()]

    def _read_setup(self):
        with open('data/setup.json') as file:
            setup = json.load(file)
        return setup

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)


class FitTransformRobustScaler(FitTransformScaler):
    abstract = False

    def run(self):
        from sklearn.preprocessing import RobustScaler
        setup = self._read_setup()
        data = self._read_tabular_data()
        transformer = RobustScaler()

        col_to_transform = list(set(data.columns) - {setup["target_column"]})
        data[col_to_transform] = transformer.fit_transform(data[col_to_transform])
        data.to_pickle(self.output().path)

    def output(self):
        return luigi.LocalTarget("data/final_preprocessed_scaled_data.pkl")


class TrainRegressionModel(luigi.Task, LuigiCombinator):
    abstract = True
    tabular_data_preprocessed_filtered_scaled = ClsParameter(tpe=FitTransformRobustScaler.return_type())
    setup = ClsParameter(tpe=WriteSetupJson.return_type())

    def requires(self):
        return [self.tabular_data_preprocessed_filtered_scaled(), self.setup()]

    def output(self):
        return luigi.LocalTarget('data/regression_model.pkl')

    def _read_setup(self):
        with open(self.input()[1].open().name) as file:
            setup = json.load(file)
        return setup

    def _read_tabular_data(self):
        return pd.read_pickle(self.input()[0].open().name)


class TrainLinearRegressionModel(TrainRegressionModel):
    abstract = False

    def run(self):
        setup = self._read_setup()
        tabular = self._read_tabular_data()
        print("TARGET:", setup["target_column"])
        print("NOW WE FIT A REGRESSION MODEL")

        X = tabular.drop(setup["target_column"], axis="columns")
        y = tabular[[setup["target_column"]]].values.ravel()
        print(y)
        print(X.shape)
        print(y.shape)
        reg = LinearRegression().fit(X, y)

        print(reg.coef_)

        with open(self.output().path, 'wb') as f:
            dump(reg, f)


class FinalNode(luigi.WrapperTask, LuigiCombinator):
    train = ClsParameter(tpe=TrainRegressionModel.return_type())

    def requires(self):
        return self.train()


if __name__ == '__main__':
    target = FinalNode.return_type()
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
    if results:
        print("Number of results", max_results)
        print("Run Pipelines")
        luigi.build(results, local_scheduler=False)  # für luigid: local_scheduler = True weglassen!
    else:
        print("No results!")
