import json
import warnings

import joblib
import luigi
import pandas as pd
from sklearn.metrics import accuracy_score

from cls_luigi.inhabitation_task import ClsParameter
from .autosklearn_task_base import AutoSklearnTask


class FeatureProvider(AutoSklearnTask):
    abstract = True


# class LoadAndSplitData(AutoSklearnTask):
class LoadAndSplitData(FeatureProvider):
    abstract = True

    def requires(self):
        return None

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl"),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl")
        }


class CategoryCoalescer(FeatureProvider):
# class CategoryCoalescer(AutoSklearnTask):
    abstract = True
    split_data = ClsParameter(tpe=LoadAndSplitData.return_type())

    component = None
    x_train = None
    x_test = None

    def requires(self):
        return self.split_data()

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def _read_split_features(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def _get_categorical_features_names(self):
        return self.x_train.select_dtypes(include=['category']).columns.tolist()

    def _save_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.component, outfile)


class CategoricalEncoder(FeatureProvider):
# class CategoricalEncoder(AutoSklearnTask):
    abstract = True
    coalesced_features = ClsParameter(tpe=FeatureProvider.return_type())
    # coalesced_features = ClsParameter(tpe=CategoryCoalescer.return_type())

    encoder = None
    x_train = None
    x_test = None
    delete_cols = []

    def requires(self):
        return self.coalesced_features()

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def _read_split_features(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def _fit_transform_encoder(self, categorical_features_names, drop_original=True, suffix=""):
        with warnings.catch_warnings(record=True) as w:
            self.encoder.fit(self.x_train[categorical_features_names])

            encoded_train_features = pd.DataFrame(
                columns=self.encoder.get_feature_names_out() + suffix,
                data=self.encoder.transform(self.x_train[categorical_features_names])
            )

            encoded_test_features = pd.DataFrame(
                columns=self.encoder.get_feature_names_out() + suffix,
                data=self.encoder.transform(self.x_test[categorical_features_names])
            )

            self.x_train = pd.concat([self.x_train, encoded_train_features], axis=1)
            self.x_test = pd.concat([self.x_test, encoded_test_features], axis=1)

            if drop_original is True:
                self.x_train.drop(categorical_features_names, axis=1, inplace=True)
                self.x_test.drop(categorical_features_names, axis=1, inplace=True)

            self._log_warnings(w)

    def _save_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.encoder, outfile)

    def _get_categorical_features_names(self):
        return self.x_train.select_dtypes(include=['category']).columns.tolist()


class NumericalImputer(AutoSklearnTask):
    abstract = True
    encoded_features = ClsParameter(tpe=FeatureProvider.return_type())
    # encoded_features = ClsParameter(tpe=CategoricalEncoder.return_type())


    imputer = None
    x_train = None
    x_test = None

    def requires(self):
        return self.encoded_features()

    def _read_split_features(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def _fit_transform_imputer(self):
        with warnings.catch_warnings(record=True) as w:
            self.imputer.fit(self.x_train)

            self.x_train = pd.DataFrame(
                columns=self.x_train.columns,
                data=self.imputer.transform(self.x_train)
            )

            self.x_test = pd.DataFrame(
                columns=self.x_test.columns,
                data=self.imputer.transform(self.x_test)
            )
            self._log_warnings(w)

    def _save_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.imputer, outfile)


class Scaler(AutoSklearnTask):
    abstract = True
    imputed_feaatures = ClsParameter(tpe=NumericalImputer.return_type())

    scaler = None
    x_train = None
    x_test = None

    def requires(self):
        return self.imputed_feaatures()

    def _read_split_imputed_features(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def fit_transform_scaler(self):
        with warnings.catch_warnings(record=True) as w:
            self.scaler.fit(self.x_train)

            self.x_train = pd.DataFrame(
                columns=self.x_train.columns,
                data=self.scaler.transform(self.x_train)
            )
            self.x_test = pd.DataFrame(
                columns=self.x_test.columns,
                data=self.scaler.transform(self.x_test)
            )
            self._log_warnings(w)

    def sava_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.scaler, outfile)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }


class FeaturePreprocessor(AutoSklearnTask):
    abstract = True
    scaled_features = ClsParameter(tpe=Scaler.return_type())
    target_values = ClsParameter(tpe=LoadAndSplitData.return_type())

    feature_preprocessor = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def requires(self):
        return {
            "scaled_features": self.scaled_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def _read_split_scaled_features(self):
        self.x_train = pd.read_pickle(self.input()["scaled_features"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["scaled_features"]["x_test"].path)

    def _read_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path)
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path)

    def sava_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.feature_preprocessor, outfile)

    def fit_transform_feature_preprocessor(self, x_and_y_required=False, handle_sparse_output=False):
        with warnings.catch_warnings(record=True) as w:
            if x_and_y_required is True:
                assert self.y_train is not None, "y_train is None!"
                self.feature_preprocessor.fit(self.x_train, self.y_train)
            else:
                self.feature_preprocessor.fit(self.x_train, self.y_train)

            if handle_sparse_output is True:
                self.x_train = pd.DataFrame.sparse.from_spmatrix(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_train)
                )

                self.x_test = pd.DataFrame.sparse.from_spmatrix(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_test)
                )

            elif handle_sparse_output is False:

                self.x_train = pd.DataFrame(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_train)
                )
                self.x_test = pd.DataFrame(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_test)
                )
            self._log_warnings(w)


class Classifier(AutoSklearnTask):
    abstract = True
    processed_features = ClsParameter(tpe=FeaturePreprocessor.return_type())
    target_values = ClsParameter(tpe=LoadAndSplitData.return_type())

    estimator = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    y_train_predict = None
    y_test_predict = None
    run_summary = {}

    def requires(self):
        return {
            "processed_features": self.processed_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "prediction": self.get_luigi_local_target_with_task_id("prediction.pkl"),
            "run_summary": self.get_luigi_local_target_with_task_id("run_summary.json"),
            "fitted_classifier": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def _read_split_processed_features(self):
        self.x_train = pd.read_pickle(self.input()["processed_features"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["processed_features"]["x_test"].path)

    def _read_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path)
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path)

    def sava_outputs(self):
        self.y_test_predict.to_pickle(self.output()["prediction"].path)

        with open(self.output()["run_summary"].path, "w") as f:
            json.dump(self.run_summary, f, indent=4)

        with open(self.output()["fitted_classifier"].path, "wb") as outfile:
            joblib.dump(self.estimator, outfile)

    def fit_predict_estimator(self):
        with warnings.catch_warnings(record=True) as w:
            self.estimator.fit(self.x_train, self.y_train)
            self.y_test_predict = pd.DataFrame(
                columns=["y_predict"],
                data=self.estimator.predict(self.x_test))

            self.y_train_predict = pd.DataFrame(
                columns=["y_predict"],
                data=self.estimator.predict(self.x_train))

            self._log_warnings(w)

    def compute_accuracy(self):
        self.run_summary["last_task"] = self.task_id,
        self.run_summary["accuracy"] = {
            "train": round(float(accuracy_score(self.y_train, self.y_train_predict)), 5),
            "test": round(float(accuracy_score(self.y_test, self.y_test_predict)), 5)
        }

    def create_run_summary(self):
        upstream_tasks = self._get_upstream_tasks()

        self.run_summary["pipeline"] = {
            "imputer": list(filter(lambda task: isinstance(task, NumericalImputer), upstream_tasks))[0].task_family,
            "scaler": list(filter(lambda task: isinstance(task, Scaler), upstream_tasks))[0].task_family,
            "feature_preprocessor": list(filter(lambda task: isinstance(task, FeaturePreprocessor), upstream_tasks))[
                0].task_family,
            "classifier": list(filter(lambda task: isinstance(task, Classifier), upstream_tasks))[0].task_family
        }

        self.compute_accuracy()

    def _get_upstream_tasks(self):
        def _get_upstream_tasks_recursively(task, upstream_list=None):
            if upstream_list is None:
                upstream_list = []

            if task not in upstream_list:
                upstream_list.append(task)

            requires = task.requires()
            if requires:
                if isinstance(requires, luigi.Task):
                    if requires not in upstream_list:
                        upstream_list.append(requires)
                    _get_upstream_tasks_recursively(requires, upstream_list)
                elif isinstance(requires, dict):
                    for key, value in requires.items():
                        if value not in upstream_list:
                            upstream_list.append(value)
                        _get_upstream_tasks_recursively(value, upstream_list)
            return upstream_list

        return _get_upstream_tasks_recursively(self)



