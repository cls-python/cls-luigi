import json
import pickle
from os.path import join as pjoin

import luigi
import pandas as pd
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler

from cls_luigi.inhabitation_task import ClsParameter, RepoMeta
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from base_task import BaseTaskClass

from global_parameters import GlobalParameters


class ReadData(BaseTaskClass):
    abstract = False

    x_train = None
    y_train = None
    x_test = None
    y_test = None
    x_infer = None
    y_infer = None
    cv_features_and_targets = None

    def _run_cv(self):
        with open(self.global_params.cv_features_and_targets_path, "rb") as f:
            self.cv_features_and_targets = pickle.load(f)

        self._save_outputs("cv")

    def _run_fit(self):
        self.x_train = pd.read_pickle(self.global_params.x_train_path)
        self.y_train = pd.read_pickle(self.global_params.y_train_path)

        self._save_outputs("fit")

    def _run_predict(self):
        self.x_test = pd.read_pickle(self.global_params.x_test_path)
        self.y_test = pd.read_pickle(self.global_params.y_test_path)

        self._save_outputs("predict")

    def _run_infer(self):
        self.x_infer = pd.read_pickle(self.global_params.x_infer_path)

        self._save_outputs("infer")

    def _save_outputs(self, mode):
        if mode == "cv":
            with open(self.output()["cv_features_and_targets"].path, "wb") as f:
                pickle.dump(self.cv_features_and_targets, f)

        if mode == "fit":
            self.x_train.to_pickle(self.output()["x_train"].path)
            self.y_train.to_pickle(self.output()["y_train"].path)

        if mode == "predict":
            self.x_test.to_pickle(self.output()["x_test"].path)
            self.y_test.to_pickle(self.output()["y_test"].path)

        if mode == "infer":
            self.x_infer.to_pickle(self.output()["x_infer"].path)

    def _cv_output(self):
        return {
            "cv_features_and_targets": self.get_luigi_local_target_with_task_id("cv_features_and_targets.pkl")
        }

    def _fit_output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl"),
        }

    def _predict_output(self):
        return {
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl")
        }

    def _infer_output(self):
        return {
            "x_infer": self.get_luigi_local_target_with_task_id("x_infer.pkl")
        }


class Scaler(BaseTaskClass):
    abstract = True
    features = ClsParameter(tpe=ReadData.return_type())

    scaler = None
    cv_transformed_features = None
    cv_fitted_components = None

    x_train = None
    x_test = None
    x_infer = None

    def requires(self):
        return self.features(mode=self.mode)

    def _run_cv(self):

        with open(self.input()["cv_features_and_targets"].path, "rb") as f:
            training_folds = pickle.load(f)

        self.cv_transformed_features = {}
        self.cv_fitted_components = {}

        for fold, features in training_folds.items():
            x_train = features["training"]["x"]
            x_validation = features["validation"]["x"]

            self.scaler = self._init_component()
            self.scaler.fit(x_train)

            self.cv_transformed_features[fold] = {
                "x_train": pd.DataFrame(self.scaler.transform(x_train),
                                        columns=x_train.columns),
                "x_validation": pd.DataFrame(self.scaler.transform(x_validation),
                                             columns=x_validation.columns)
            }
            self.cv_fitted_components[fold] = self.scaler
            self._save_outputs("cv")
            self.scaler = None

    def _run_fit(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)

        self.scaler = self._init_component()
        self.scaler.fit(self.x_train)

        self.x_train = pd.DataFrame(self.scaler.transform(self.x_train),
                                    columns=self.x_train.columns)

        self._save_outputs("fit")

    def _run_predict(self):
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

        scaler = self._load_component()

        self.x_test = pd.DataFrame(scaler.transform(self.x_test),
                                   columns=self.x_test.columns)
        self._save_outputs("predict")

    def _run_infer(self):
        self.x_infer = pd.read_pickle(self.input()["x_infer"].path)

        scaler = self._load_component()

        self.x_infer = pd.DataFrame(scaler.transform(self.x_infer),
                                    columns=self.x_infer.columns)
        self._save_outputs("infer")

    def _save_outputs(self, mode):

        if mode == "cv":
            with open(self.output()["cv_transformed_features"].path, "wb") as f:
                pickle.dump(self.cv_transformed_features, f)

            with open(self.output()["cv_fitted_components"].path, "wb") as f:
                pickle.dump(self.cv_fitted_components, f)

        if mode == "fit":
            self.x_train.to_pickle(self.output()["x_train"].path)

            with open(self.output()["fitted_component"].path, "wb") as f:
                pickle.dump(self.scaler, f)

        if mode == "predict":
            self.x_test.to_pickle(self.output()["x_test"].path)

        if mode == "infer":
            self.x_infer.to_pickle(self.output()["x_infer"].path)

    def _cv_output(self):
        return {
            "cv_transformed_features": self.get_luigi_local_target_with_task_id(
                "cv_transformed_features.pkl"),
            "cv_fitted_components": self.get_luigi_local_target_with_task_id("cv_fitted_components.pkl")
        }

    def _fit_output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def _predict_output(self):
        return {
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
        }

    def _infer_output(self):
        return {
            "x_infer": self.get_luigi_local_target_with_task_id("x_infer.pkl")
        }


class SKLStandardScaler(Scaler):
    abstract = False

    def _init_component(self):
        return StandardScaler()


class SKLRobustScaler(Scaler):
    abstract = False

    def _init_component(self):
        return RobustScaler()


class Classifier(BaseTaskClass):
    abstract = True
    target_values = ClsParameter(tpe=ReadData.return_type())
    transformed_features = ClsParameter(tpe=Scaler.return_type())

    classifier = None
    cv_predictions = None
    cv_scores = None
    cv_fitted_components = None

    scores = None
    x_train = None
    y_train = None
    train_prediction = None

    x_test = None
    y_test = None

    infer_prediction = None

    def requires(self):
        return {
            "target_values": self.target_values(mode=self.mode),
            "transformed_features": self.transformed_features(mode=self.mode)
        }

    @staticmethod
    def _compute_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def _run_cv(self):

        with open(self.input()["target_values"]["cv_features_and_targets"].path, "rb") as f:
            cv_target = pickle.load(f)

        with open(self.input()["transformed_features"]["cv_transformed_features"].path, "rb") as f:
            cv_transformed_features = pickle.load(f)

        self.cv_predictions = {}
        self.cv_scores = {}
        self.cv_fitted_components = {}

        for fold, features in cv_transformed_features.items():
            x_train = features["x_train"]
            x_validation = features["x_validation"]

            y_train = cv_target[fold]["training"]["y"]
            y_validation = cv_target[fold]["validation"]["y"]

            self.classifier = self._init_component()
            self.classifier.fit(x_train, y_train)

            train_prediction = self.classifier.predict(x_train)
            validation_prediction = self.classifier.predict(x_validation)

            self.cv_predictions[fold] = {
                "train_prediction": train_prediction,
                "validation_prediction": validation_prediction,
            }

            self.cv_scores[fold] = {
                "train_accuracy": self._compute_accuracy(y_train, train_prediction),
                "validation_accuracy": self._compute_accuracy(y_validation, validation_prediction)
            }
            self.cv_fitted_components[fold] = self.classifier
            self.classifier = None

        mean_train_accuracy = sum([v["train_accuracy"] for v in self.cv_scores.values()]) / len(
            self.cv_scores)
        mean_validation_accuracy = sum([v["validation_accuracy"] for v in self.cv_scores.values()]) / len(
            self.cv_scores)

        self.cv_scores["mean_train_accuracy"] = mean_train_accuracy
        self.cv_scores["mean_validation_accuracy"] = mean_validation_accuracy

        self._save_outputs("cv")

    def _run_fit(self):
        self.scores = {}
        self.x_train = pd.read_pickle(self.input()["transformed_features"]["x_train"].path)
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path)

        self.classifier = self._init_component()
        self.classifier.fit(self.x_train, self.y_train)

        self.train_prediction = pd.DataFrame(
            self.classifier.predict(self.x_train),
            columns=["prediction"]
        )

        train_accuracy = self._compute_accuracy(self.y_train, self.train_prediction)
        self.scores["train_accuracy"] = train_accuracy

        self._save_outputs("fit")
        self.classifier = None

    def _run_predict(self):
        self.scores = {}
        self.x_test = pd.read_pickle(self.input()["transformed_features"]["x_test"].path)
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path)

        self.classifier = self._load_component()

        self.test_prediction = pd.DataFrame(
            self.classifier.predict(self.x_test),
            columns=["prediction"]
        )

        train_accuracy = self._compute_accuracy(self.y_test, self.test_prediction)
        self.scores["test_accuracy"] = train_accuracy

        self._save_outputs("predict")
        self.classifier = None

    def _run_infer(self):
        self.x_infer = pd.read_pickle(self.input()["transformed_features"]["x_infer"].path)

        self.classifier = self._load_component()

        self.infer_prediction = pd.DataFrame(
            self.classifier.predict(self.x_infer),
            columns=["prediction"]
        )

        self._save_outputs("infer")
        self.classifier = None

    def _save_outputs(self, mode):
        if mode == "cv":
            with open(self._cv_output()["cv_predictions"].path, "wb") as f:
                pickle.dump(self.cv_predictions, f)

            with open(self._cv_output()["cv_fitted_components"].path, "wb") as f:
                pickle.dump(self.cv_fitted_components, f)

            with open(self._cv_output()["cv_scores"].path, "w") as f:
                json.dump(self.cv_scores, f, indent=5)

        if mode == "fit":
            pass
            self.train_prediction.to_pickle(self._fit_output()["train_prediction"].path)

            with open(self._fit_output()["fitted_component"].path, "wb") as f:
                pickle.dump(self.classifier, f)

            with open(self._fit_output()["train_accuracy"].path, "w") as f:
                json.dump(self.scores, f, indent=5)

        if mode == "predict":
            self.test_prediction.to_pickle(self._predict_output()["test_prediction"].path)

            with open(self._predict_output()["test_accuracy"].path, "w") as f:
                json.dump(self.scores, f, indent=5)

        if mode == "infer":
            self.infer_prediction.to_pickle(self._infer_output()["infer_prediction"].path)

    def _cv_output(self):
        return {
            "cv_predictions": self.get_luigi_local_target_with_task_id("cv_predictions.pkl"),
            "cv_fitted_components": self.get_luigi_local_target_with_task_id("cv_fitted_components.pkl"),
            "cv_scores": self.get_luigi_local_target_with_task_id("cv_scores.json")
        }

    def _fit_output(self):
        return {
            "train_prediction": self.get_luigi_local_target_with_task_id("train_prediction.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl"),
            "train_accuracy": self.get_luigi_local_target_with_task_id("train_accuracy.json")
        }

    def _predict_output(self):
        return {
            "test_prediction": self.get_luigi_local_target_with_task_id("test_prediction.pkl"),
            "test_accuracy": self.get_luigi_local_target_with_task_id("test_accuracy.json")

        }

    def _infer_output(self):
        return {
            "infer_prediction": self.get_luigi_local_target_with_task_id("infer_prediction.pkl")
        }


class SKLDecisionTreeClassifier(Classifier):
    abstract = False

    def _init_component(self):
        return tree.DecisionTreeClassifier()


if __name__ == "__main__":

    ds_dir = "datasets/breast_cancer"
    gp = GlobalParameters()
    gp.dataset_name = "breast_cancer"
    gp.x_train_path = pjoin(ds_dir, "x_train.pkl")
    gp.y_train_path = pjoin(ds_dir, "y_train.pkl")
    gp.x_test_path = pjoin(ds_dir, "x_test.pkl")
    gp.y_test_path = pjoin(ds_dir, "y_test.pkl")
    gp.x_infer_path = pjoin(ds_dir, "x_infer.pkl")
    gp.cv_features_and_targets_path = pjoin(ds_dir, "cv_features_and_targets.pkl")

    target_values = Classifier.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Building Repository")

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines")

    inhabitation_result = fcl.inhabit(target_values)
    print("Enumerating results")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if actual > 0:
        max_results = actual

    validator = UniqueTaskPipelineValidator(
        [Scaler,
         Classifier])

    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Running Pipelines")

        for ix, r in enumerate(results):
            results[ix].mode = ("cv", "fit", "predict", "infer")

        luigi_run_result = luigi.build(results,
                                       local_scheduler=True,
                                       # detailed_summary=True
                                       )
