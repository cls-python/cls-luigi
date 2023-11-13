import sys
import json
import logging
import warnings

import joblib
import luigi
import numpy as np
import pandas as pd

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomTreesEmbedding, AdaBoostClassifier, \
    HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, chi2, \
    GenericUnivariateSelect
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, \
    Normalizer, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from cls_luigi.inhabitation_task import RepoMeta, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from autosklearn_task import AutoSklearnTask

logging.captureWarnings(True)
from luigi.execution_summary import execution_summary

sys.path.append('..')
sys.path.append('../..')

execution_summary.summary_length = 10000

SEED = 123
NJOBS = 1
TIMEOUT = 10  # seconds for SVM Classifiers
DATASET_NAME = None  # pc4 # phoneme #sylvine #


class LoadAndSplitCSV(AutoSklearnTask):
    abstract = False

    def run(self):
        df = pd.read_csv("datasets/{}.csv".format(DATASET_NAME))

        x_train, x_test, y_train, y_test = train_test_split(
            df.drop("target", axis=1), df["target"], test_size=0.33, random_state=SEED)

        x_train.to_pickle(self.output()["x_train"].path)
        x_test.to_pickle(self.output()["x_test"].path)
        y_train.to_pickle(self.output()["y_train"].path)
        y_test.to_pickle(self.output()["y_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl", dataset_name=DATASET_NAME),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl", dataset_name=DATASET_NAME),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl", dataset_name=DATASET_NAME),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl", dataset_name=DATASET_NAME)
        }


class NumericalImputation(AutoSklearnTask):
    abstract = True
    split_data = ClsParameter(tpe=LoadAndSplitCSV.return_type())

    imputer = None
    x_train = None
    x_test = None

    def requires(self):
        return self.split_data()

    def _read_split_features_from_input(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl", dataset_name=DATASET_NAME),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl", dataset_name=DATASET_NAME),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl",
                                                                         dataset_name=DATASET_NAME)
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


class SKLSimpleImpute(NumericalImputation):
    abstract = False

    def run(self):
        self._read_split_features_from_input()

        self.imputer = SimpleImputer(
            strategy="mean",
            copy=False
        )

        self._fit_transform_imputer()
        self._save_outputs()


class Scaling(AutoSklearnTask):
    abstract = True
    imputed_data = ClsParameter(tpe=NumericalImputation.return_type())

    scaler = None
    x_train = None
    x_test = None

    def requires(self):
        return self.imputed_data()

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
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl", dataset_name=DATASET_NAME),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl", dataset_name=DATASET_NAME),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl",
                                                                         dataset_name=DATASET_NAME)
        }


class NoScaling(Scaling):
    abstract = False

    def output(self):
        return {
            "x_train": self.input()["x_train"],
            "x_test": self.input()["x_test"],
            "fitted_component": self.input()["fitted_component"]
        }


class SKLStandardScaler(Scaling):
    abstract = False

    def run(self):
        self.scaler = StandardScaler(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLRobustScaler(Scaling):
    abstract = False

    def run(self):
        self.scaler = RobustScaler(
            copy=False,
            quantile_range=(0.25, 0.75)
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLMinMaxScaler(Scaling):
    abstract = False

    def run(self):
        self.scaler = MinMaxScaler(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLQuantileTransformer(Scaling):
    abstract = False

    def run(self):
        self.scaler = QuantileTransformer(
            copy=False,
            n_quantiles=1000,
            output_distribution="uniform",
            random_state=SEED
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLPowerTransformer(Scaling):
    abstract = False

    def run(self):
        self.scaler = PowerTransformer(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLNormalizer(Scaling):
    abstract = False

    def run(self):
        self.scaler = Normalizer(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


# Feature Preprocessing
class FeaturePreprocessor(AutoSklearnTask):
    abstract = True
    scaled_features = ClsParameter(tpe=Scaling.return_type())
    target_values = ClsParameter(tpe=LoadAndSplitCSV.return_type())

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
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl", dataset_name=DATASET_NAME),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl", dataset_name=DATASET_NAME),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl",
                                                                         dataset_name=DATASET_NAME)
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


class NoFeaturePreprocessor(FeaturePreprocessor):
    abstract = False

    def output(self):
        return {
            "x_train": self.input()["scaled_features"]["x_train"],
            "x_test": self.input()["scaled_features"]["x_test"],
            "fitted_component": self.input()["scaled_features"]["fitted_component"]
        }


class SKLSelectFromExtraTrees(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()
        self._read_split_target_values()

        estimator = ExtraTreesClassifier(
            n_estimators=100,
            criterion="gini",
            max_features=0.5,
            max_depth=None,
            max_leaf_nodes=None,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            min_impurity_decrease=0.0,
            bootstrap=False,
            random_state=SEED,
            oob_score=False,
            n_jobs=NJOBS,
            verbose=0
        )

        estimator.fit(self.x_train, self.y_train)
        self.feature_preprocessor = SelectFromModel(
            estimator=estimator,
            threshold="mean",
            prefit=True
        )
        self.fit_transform_feature_preprocessor(x_and_y_required=True)
        self.sava_outputs()


class SKLSelectFromLinearSVC(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()
        self._read_split_target_values()

        estimator = LinearSVC(
            penalty="l1",
            loss="squared_hinge",
            dual=False,
            tol=1e-4,
            C=1.0,
            multi_class="ovr",
            fit_intercept=True,
            intercept_scaling=1,
            random_state=SEED
        )
        estimator.fit(self.x_train, self.y_train)
        self.feature_preprocessor = SelectFromModel(
            estimator=estimator,
            threshold="mean",
            prefit=True
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=True)
        self.sava_outputs()


class SKLFastICA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            self._read_split_scaled_features()

            self.feature_preprocessor = FastICA(
                # n_components=default_hyperparameters["n_components"],
                algorithm="parallel",
                whiten=False,
                fun="logcosh",
                random_state=SEED
            )

            self.fit_transform_feature_preprocessor(x_and_y_required=False)
            self.sava_outputs()


class SKLFeatureAgglomeration(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = FeatureAgglomeration(
            n_clusters=min(25, self.x_train.shape[1]),
            affinity="euclidean",
            linkage="ward",
            pooling_func=np.mean
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()


class SKLKernelPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = KernelPCA(
            n_components=100,
            kernel="rbf",
            gamma=0.1,
            degree=3,
            coef0=0,
            remove_zero_eig=True,
            random_state=SEED,
        )
        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()


class SKLRBFSampler(FeaturePreprocessor):
    # aka kitchen sinks
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = RBFSampler(
            gamma=1.0,
            n_components=100,
            random_state=SEED
        )
        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()


class SKLNystroem(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = Nystroem(
            kernel="rbf",
            gamma=1.0,
            coef0=0,
            degree=3,
            n_components=100,
            random_state=SEED

        )

        # self.x_train[self.x_train < 0] = 0.0
        # self.x_test[self.x_test < 0] = 0.0

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()


class SKLPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = PCA(
            n_components=0.9999,
            whiten=False,
            copy=True,
            random_state=SEED,
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()


class SKLPolynomialFeatures(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = PolynomialFeatures(
            degree=2,
            interaction_only=False,
            include_bias=True
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()


class SKLRandomTreesEmbedding(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = RandomTreesEmbedding(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0,  # TODO: check
            max_leaf_nodes=None,
            n_jobs=NJOBS,
            random_state=SEED,
            sparse_output=False
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False, handle_sparse_output=False)
        self.sava_outputs()


class SKLSelectPercentile(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()
        self._read_split_target_values()

        self.feature_preprocessor = SelectPercentile(
            score_func=chi2,
            percentile=50
        )
        # self.x_train[self.x_train < 0] = 0.0
        # self.x_test[self.x_test < 0] = 0.0

        self.fit_transform_feature_preprocessor(x_and_y_required=True)
        self.sava_outputs()


class SKLSelectRates(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()
        self._read_split_target_values()

        self.feature_preprocessor = GenericUnivariateSelect(
            score_func=chi2,
            mode="fpr",
            param=0.1
        )
        self.fit_transform_feature_preprocessor(x_and_y_required=True)
        self.sava_outputs()


# classifiers
class Classifier(AutoSklearnTask):
    abstract = True
    processed_features = ClsParameter(tpe=FeaturePreprocessor.return_type())
    target_values = ClsParameter(tpe=LoadAndSplitCSV.return_type())

    estimator = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    y_train_predict = None
    y_test_predict = None
    performance_scores = {}

    def requires(self):
        return {
            "processed_features": self.processed_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "prediction": self.get_luigi_local_target_with_task_id("prediction.pkl", dataset_name=DATASET_NAME),
            "scores": self.get_luigi_local_target_with_task_id("score.json", dataset_name=DATASET_NAME),
            "fitted_classifier": self.get_luigi_local_target_with_task_id("fitted_component.pkl",
                                                                          dataset_name=DATASET_NAME)
        }

    def _read_split_processed_features(self):
        self.x_train = pd.read_pickle(self.input()["processed_features"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["processed_features"]["x_test"].path)

    def _read_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path)
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path)

    def sava_outputs(self):
        self.y_test_predict.to_pickle(self.output()["prediction"].path)

        with open(self.output()["scores"].path, "w") as f:
            json.dump(self.performance_scores, f, indent=4)

        with open(self.output()["fitted_classifier"].path, 'wb') as outfile:
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
        self.performance_scores["name"] = self.task_id,
        self.performance_scores["accuracy"] = {
            "train": round(float(accuracy_score(self.y_train, self.y_train_predict)), 5),
            "test": round(float(accuracy_score(self.y_test, self.y_test_predict)), 5)
        }


class SKLAdaBoost(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=SEED)

        self.estimator = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=50,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=SEED
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLDecisionTree(Classifier):
    abstract = False

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        num_features = self.x_train.shape[1]

        max_depth_factor = max(
            1, int(np.round(0.5 * num_features, 0))
        )

        self.estimator = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth_factor,  #
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            random_state=SEED,
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLExtraTrees(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        max_features = int(self.x_train.shape[1] ** 0.5)
        if max_features == 0:
            max_features = "sqrt"
        self.estimator = ExtraTreesClassifier(
            criterion="gini",
            max_features=max_features,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=False,
            random_state=SEED
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLGaussianNaiveBayes(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = GaussianNB()

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLSGD(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            l1_ratio=0.15,
            fit_intercept=True,
            tol=1e-4,
            epsilon=1e-4,
            learning_rate="invscaling",
            eta0=0.01,
            power_t=0.5,
            average=False,
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLGradientBoosting(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=None,
            max_leaf_nodes=31,
            max_bins=255,
            l2_regularization=1e-10,
            early_stopping=False,
            tol=1e-7,
            scoring="loss",
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=SEED,
            warm_start=True
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLKNearestNeighbors(Classifier):
    abstract = False

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = KNeighborsClassifier(
            n_neighbors=1,
            weights="uniform",
            p=2
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLLinearDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = LinearDiscriminantAnalysis(
            shrinkage=None,
            solver="svd",
            tol=1e-1,
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLLinearSVC(Classifier):
    abstract = False
    worker_timeout = TIMEOUT

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            tol=1e-4,
            C=1.0,
            multi_class="ovr",
            fit_intercept=True,
            intercept_scaling=1,
            random_state=SEED
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLKernelSVC(Classifier):
    abstract = False
    worker_timeout = TIMEOUT

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = SVC(
            C=1.0,
            kernel="rbf",
            degree=3,
            gamma=0.1,
            coef0=0.0,
            shrinking=True,
            tol=1e-3,
            max_iter=-1,
            random_state=SEED,
            class_weight=None,
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKMultinomialNB(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        # self.x_train[self.x_train < 0] = 0.0
        # self.x_test[self.x_test < 0] = 0.0

        self.estimator = MultinomialNB(
            alpha=1,
            fit_prior=True
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLPassiveAggressive(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = PassiveAggressiveClassifier(
            C=1.0,
            fit_intercept=True,
            loss="hinge",
            tol=1e-4,
            average=False,
            shuffle=True,
            random_state=SEED,
            warm_start=True
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLQuadraticDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = QuadraticDiscriminantAnalysis(
            reg_param=0.0,
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLRandomForest(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        max_features = int(self.x_train.shape[1] ** 0.5)

        self.estimator = RandomForestClassifier(
            criterion="gini",
            max_features=max_features,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            random_state=SEED,
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLBernoulliNB(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = BernoulliNB(
            alpha=1.0,
            fit_prior=True,

        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class GenerateAndUpdateLeaderboard(AutoSklearnTask):
    abstract = False
    predictions_and_scores = ClsParameter(tpe=Classifier.return_type())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_scores = None
        self.done = False

        self.df_leaderboard = None

    def complete(self):
        return self.done

    def requires(self):
        return self.predictions_and_scores()

    def output(self):
        # return luigi.LocalTarget("leaderboard.csv")
        return self.get_luigi_local_target_without_task_id("01_leaderboard.csv", dataset_name=DATASET_NAME)

    def _read_accuracy_scores(self):
        with open(self.input()["scores"].path, "r") as f:
            self.accuracy_scores = json.load(f)

    def run(self):
        self._read_accuracy_scores()
        if self.output().exists():
            self.df_leaderboard = pd.read_csv(self.output().path)
        else:
            self.df_leaderboard = pd.DataFrame(
                columns=["imputation", "scaling", "Feature_preprocessing", "classification", "train_score",
                         "test_score"]
            )

        upstream_tasks = self._get_upstream_tasks()
        imputer = list(filter(lambda task: isinstance(task, NumericalImputation), upstream_tasks))[0]
        scaler = list(filter(lambda task: isinstance(task, Scaling), upstream_tasks))[0]
        feature_preprocessor = list(filter(lambda task: isinstance(task, FeaturePreprocessor), upstream_tasks))[0]
        classifier = list(filter(lambda task: isinstance(task, Classifier), upstream_tasks))[0]

        new_row = [imputer.task_family,
                   scaler.task_family,
                   feature_preprocessor.task_family,
                   classifier.task_family,
                   round(self.accuracy_scores["accuracy"]["train"], 5),
                   round(self.accuracy_scores["accuracy"]["test"], 5)
                   ]
        self.df_leaderboard.loc[len(self.df_leaderboard)] = new_row

        self.df_leaderboard.sort_values(by=["train_score", "test_score"], ascending=False, inplace=True)
        self.df_leaderboard.drop_duplicates(inplace=True)
        self.df_leaderboard.to_csv(self.output().path, index=False)

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


def main():
    # from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = GenerateAndUpdateLeaderboard.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Building Repository")

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    validator = UniqueTaskPipelineValidator(
        [NumericalImputation, Scaling, FeaturePreprocessor, Classifier])

    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    if results:
        # DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Running Pipelines")
        luigi_run_result = luigi.build(results, local_scheduler=True, detailed_summary=True,
                                       logging_conf_file="logging.conf")
        print(luigi_run_result.summary_text)
        print("Done!")
    else:
        print("No results!")


if __name__ == '__main__':
    logger = logging.getLogger("luigi-root")

    for i in ['pc4', 'MiniBooNE', 'APSFailure', 'phoneme', 'jasmine', 'kc1', 'wilt', 'sylvine']:
        print("Dataset {}".format(i))
        DATASET_NAME = i
        main()
        logger.debug("\n{}\n{} This was dataset: {} {}\n{}\n".format(
            "*" * 150,
            "*" * 65,
            DATASET_NAME,
            "*" * (65 - len(DATASET_NAME)),
            "*" * 150))
