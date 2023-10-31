import sys

sys.path.append("../")
sys.path.append("../../")
import joblib
import luigi
import numpy as np
import pandas as pd
from ConfigSpace.conditions import EqualsCondition, InCondition, NotEqualsCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenInClause, ForbiddenEqualsClause
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomTreesEmbedding, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, chi2, f_classif, mutual_info_classif, \
    GenericUnivariateSelect
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, \
    Normalizer, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from cls_luigi.inhabitation_task import RepoMeta, ClsParameter
from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, UnParametrizedHyperparameter

from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

SEED = 123
NJOBS = 1


# Load and Split --> one-Hot Encoder --> Imputation --> Scaling --> Feature Preprocessing --> Classification

from autosklearn_task import AutoSklearnTask


class LoadAndSplitCSV(AutoSklearnTask):
    abstract = False

    def run(self):
        df = pd.read_csv("data.csv")

        x_train, x_test, y_train, y_test = train_test_split(
            df.drop("target", axis=1), df["target"], test_size=0.33, random_state=SEED)

        x_train.to_pickle(self.output()["x_train"].path)
        x_test.to_pickle(self.output()["x_test"].path)
        y_train.to_pickle(self.output()["y_train"].path)
        y_test.to_pickle(self.output()["y_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl"),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl")
        }


# Imputation
class SKLSimpleImpute(AutoSklearnTask):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/imputation/numerical_imputation.py
    """
    abstract = False
    split_data = ClsParameter(tpe=LoadAndSplitCSV.return_type())

    def requires(self):
        return self.split_data()

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        strategy = CategoricalHyperparameter(name="strategy", choices=["mean", "median", "most_frequent"],
                                             default_value="mean")
        copy = Constant(name="copy", value="False")
        cs.add_hyperparameters([strategy, copy])
        return cs

    def get_split_data(self):
        x_train = pd.read_pickle(self.input()["x_train"].path)
        x_test = pd.read_pickle(self.input()["x_test"].path)
        y_train = pd.read_pickle(self.input()["y_train"].path)
        y_test = pd.read_pickle(self.input()["y_test"].path)

        return x_train, x_test, y_train, y_test

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        x_train, x_test, y_train, y_test = self.get_split_data()

        imputer = SimpleImputer(
            strategy=default_hyperparameters["strategy"],
            copy=default_hyperparameters["copy"]
        )
        imputer.fit(x_train)

        x_train = pd.DataFrame(
            columns=x_train.columns,
            data=imputer.transform(x_train)
        )

        x_test = pd.DataFrame(
            columns=x_test.columns,
            data=imputer.transform(x_test)
        )

        x_train.to_pickle(self.output()["x_train"].path)
        x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(imputer, outfile)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }


# Scalers
class Scaling(AutoSklearnTask):
    abstract = True
    imputed_data = ClsParameter(tpe=SKLSimpleImpute.return_type())

    scaler = None
    x_train = None
    x_test = None

    def requires(self):
        return {
            "imputed_data": self.imputed_data()
        }

    def load_split_imputed_features(self):
        self.x_train = pd.read_pickle(self.input()["imputed_data"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["imputed_data"]["x_test"].path)

    def fit_transform_scaler(self):
        self.scaler.fit(self.x_train)

        self.x_train = pd.DataFrame(
            columns=self.x_train.columns,
            data=self.scaler.transform(self.x_train)
        )
        self.x_test = pd.DataFrame(
            columns=self.x_test.columns,
            data=self.scaler.transform(self.x_test)
        )

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


class NoScaling(Scaling):
    abstract = False

    def output(self):
        return {
            "x_train": self.input()["imputed_data"]["x_train"],
            "x_test": self.input()["imputed_data"]["x_test"],
            "fitted_component": self.input()["imputed_data"]["fitted_component"]
        }


class SKLStandardScaler(Scaling):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/rescaling/standardize.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        copy = Constant(name="copy", value="False")
        cs.add_hyperparameter(copy)
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        self.scaler = StandardScaler(
            copy=default_hyperparameters["copy"]
        )
        self.load_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLRobustScaler(Scaling):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/rescaling/robust_scaler.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        q_min = UniformFloatHyperparameter(name="q_min", lower=.001, upper=.3, default_value=0.25)
        q_max = UniformFloatHyperparameter(name="q_max", lower=0.7, upper=0.999,
                                           default_value=0.75)  # q min and max are used together in the quantile_range
        copy = Constant(name="copy", value="False")
        cs.add_hyperparameters([q_min, q_max, copy])

        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        self.scaler = RobustScaler(
            copy=default_hyperparameters["copy"],
            quantile_range=(default_hyperparameters["q_min"], default_hyperparameters["q_max"])
        )
        self.load_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLMinMaxScaler(Scaling):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/rescaling/minmax.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        copy = Constant(name="copy", value="False")
        cs.add_hyperparameter(copy)
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        self.scaler = MinMaxScaler(
            copy=default_hyperparameters["copy"]
        )
        self.load_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLQuantileTransformer(Scaling):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/rescaling/quantile_transformer.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        n_quantiles = UniformIntegerHyperparameter(
            name="n_quantiles", lower=10, upper=2000, default_value=1000
        )
        output_distribution = CategoricalHyperparameter(
            name="output_distribution", choices=["normal", "uniform"], default_value="uniform"  # from sklearn
        )
        random_state = Constant(name="random_state", value=SEED)
        copy = Constant(name="copy", value="False")

        cs.add_hyperparameters([copy, n_quantiles, output_distribution, random_state])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        self.scaler = QuantileTransformer(
            copy=default_hyperparameters["copy"],
            n_quantiles=default_hyperparameters["n_quantiles"],
            output_distribution=default_hyperparameters["output_distribution"],
            random_state=default_hyperparameters["random_state"]
        )
        self.load_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLPowerTransformer(Scaling):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/rescaling/power_transformer.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        copy = Constant(name="copy", value="False")
        cs.add_hyperparameter(copy)
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        self.scaler = PowerTransformer(
            copy=default_hyperparameters["copy"],
        )
        self.load_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()


class SKLNormalizer(Scaling):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/data_preprocessing/rescaling/normalize.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        copy = Constant(name="copy", value="False")
        cs.add_hyperparameter(copy)
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()

        self.scaler = Normalizer(
            copy=default_hyperparameters["copy"],
        )
        self.load_split_imputed_features()
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
            "scaled_data": self.scaled_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def load_split_scaled_features(self):
        self.x_train = pd.read_pickle(self.input()["scaled_data"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["scaled_data"]["x_test"].path)

    def load_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path)
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path)

    def sava_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.feature_preprocessor, outfile)

    def fit_transform_feature_preprocessor(self, x_and_y=False):

        if x_and_y is True:
            assert self.y_train is not None, "y_train is None!"
            self.feature_preprocessor.fit(self.x_train, self.y_train)
        else:
            self.feature_preprocessor.fit(self.x_train, self.y_train)

        self.x_train = pd.DataFrame(
            columns=self.feature_preprocessor.get_feature_names_out(),
            data=self.feature_preprocessor.transform(self.x_train)
        )
        self.x_test = pd.DataFrame(
            columns=self.feature_preprocessor.get_feature_names_out(),
            data=self.feature_preprocessor.transform(self.x_test)
        )


class NoFeaturePreprocessor(FeaturePreprocessor):
    abstract = False

    def output(self):
        return {
            "x_train": self.input()["scaled_data"]["x_train"],
            "x_test": self.input()["scaled_data"]["x_test"],
            "fitted_component": self.input()["scaled_data"]["fitted_component"]
        }


class SKLSelectFromExtraTrees(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/extra_trees_preproc_for_classification.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        # ExtraTreesClassifier hyperparameters
        n_estimators = Constant(
            name="n_estimators", value=100)
        criterion = CategoricalHyperparameter(
            name="criterion", choices=["gini", "entropy"], default_value="gini")
        max_depth = UnParametrizedHyperparameter(
            name="max_depth", value="None")
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1)
        bootstrap = CategoricalHyperparameter(
            name="bootstrap", choices=["True", "False"], default_value="False")
        max_features = UniformFloatHyperparameter(
            name="max_features", lower=0.1, upper=1.0, default_value=0.5)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        min_impurity_decrease = UnParametrizedHyperparameter(
            name="min_impurity_decrease", value=0.0)
        oob_score = Constant(
            name="oob_score", value="False")
        min_weight_fraction_leaf = UnParametrizedHyperparameter(
            name="min_weight_fraction_leaf", value=0.0)
        n_jobs = Constant(
            name="n_jobs", value=NJOBS)
        verbose = Constant(
            name="verbose", value=0)
        random_state = Constant(
            name="random_state", value=SEED)
        class_weight = Constant(
            name="class_weight", value="None")
        cs.add_hyperparameters(
            [n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, bootstrap, max_features,
             max_leaf_nodes, min_impurity_decrease, oob_score, min_weight_fraction_leaf, n_jobs, verbose, random_state,
             class_weight])

        # selector function hyperparameters
        sample_weight = Constant(
            name="sample_weight", value="None")
        threshold = Constant(
            name="threshold", value="mean")
        prefit = Constant(
            name="prefit", value="True")
        cs.add_hyperparameters([sample_weight, threshold, prefit])

        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()
        self.load_split_target_values()

        default_hyperparameters["max_features"] = int(
            self.x_train.shape[1] ** float(default_hyperparameters["max_features"]))

        estimator = ExtraTreesClassifier(
            n_estimators=default_hyperparameters["n_estimators"],
            criterion=default_hyperparameters["criterion"],
            max_depth=default_hyperparameters["max_depth"],
            min_samples_split=default_hyperparameters["min_samples_split"],
            min_samples_leaf=default_hyperparameters["min_samples_leaf"],
            bootstrap=default_hyperparameters["bootstrap"],
            max_features=default_hyperparameters["max_features"],
            max_leaf_nodes=default_hyperparameters["max_leaf_nodes"],
            min_impurity_decrease=default_hyperparameters["min_impurity_decrease"],
            oob_score=default_hyperparameters["oob_score"],
            # min_weight_fraction_leaf=default_hyperparameters["min_weight_fraction_leaf"], # also not passed on in Autosklearn
            n_jobs=default_hyperparameters["n_jobs"],
            verbose=default_hyperparameters["verbose"],
            random_state=default_hyperparameters["random_state"],
            class_weight=default_hyperparameters["class_weight"]
        )

        estimator.fit(self.x_train, self.y_train, sample_weight=None)
        self.feature_preprocessor = SelectFromModel(
            estimator=estimator,
            threshold=default_hyperparameters["threshold"],
            prefit=default_hyperparameters["prefit"]
        )

        self.fit_transform_feature_preprocessor(x_and_y=True)
        self.sava_outputs()


class SKLSelectFromLinearSVC(FeaturePreprocessor):
    """
https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/liblinear_svc_preprocessor.py    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        # LinearSVC hyperparameters
        penalty = Constant(name="penalty", value="l1")
        loss = CategoricalHyperparameter(
            name="loss", choices=["hinge", "squared_hinge"], default_value="squared_hinge")
        dual = Constant(name="dual", value="False")
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, log=True, default_value=1e-4)
        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default_value=1.0)
        multi_class = Constant(name="multi_class", value="ovr")
        fit_intercept = Constant(name="fit_intercept", value="True")
        intercept_scaling = Constant(name="intercept_scaling", value=1)

        random_state = Constant(name="random_state", value=SEED)
        class_weight = Constant(name="class_weight", value="None")

        cs.add_hyperparameters(
            [penalty, loss, dual, tol, C, multi_class, fit_intercept, intercept_scaling, random_state, class_weight])

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"), ForbiddenEqualsClause(loss, "hinge")
        )
        cs.add_forbidden_clause(penalty_and_loss)

        # SelectFromModel hyperparameters
        threshold = Constant(name="threshold", value="mean")
        prefit = Constant(name="prefit", value="True")
        cs.add_hyperparameters([threshold, prefit])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()
        self.load_split_target_values()

        estimator = LinearSVC(
            penalty=default_hyperparameters["penalty"],
            loss=default_hyperparameters["loss"],
            dual=default_hyperparameters["dual"],
            tol=default_hyperparameters["tol"],
            C=default_hyperparameters["C"],
            multi_class=default_hyperparameters["multi_class"],
            fit_intercept=default_hyperparameters["fit_intercept"],
            intercept_scaling=default_hyperparameters["intercept_scaling"],
            random_state=default_hyperparameters["random_state"]
        )

        estimator.fit(self.x_train, self.y_train)
        self.feature_preprocessor = SelectFromModel(
            estimator=estimator,
            threshold=default_hyperparameters["threshold"],
            prefit=default_hyperparameters["prefit"]
        )

        self.fit_transform_feature_preprocessor(x_and_y=True)
        self.sava_outputs()


class SKLFastICA(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/fast_ica.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=10, upper=2000, default_value=100
        )
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["parallel", "deflation"], default_value="parallel"
        )
        whiten = CategoricalHyperparameter(name="whiten", choices=[False, True], default_value=False)
        fun = CategoricalHyperparameter(name="fun", choices=["logcosh", "exp", "cube"], default_value="logcosh")
        random_state = Constant(name="random_state", value=SEED)
        cs.add_hyperparameters([n_components, algorithm, whiten, fun, random_state])

        cs.add_condition(EqualsCondition(n_components, whiten, True))

        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        self.feature_preprocessor = FastICA(
            # n_components=default_hyperparameters["n_components"],
            algorithm=default_hyperparameters["algorithm"],
            whiten=default_hyperparameters["whiten"],
            fun=default_hyperparameters["fun"],
            random_state=default_hyperparameters["random_state"]
        )

        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLFeatureAgglomeration(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/feature_agglomeration.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        n_clusters = UniformIntegerHyperparameter(name="n_clusters", lower=2, upper=400, default_value=25)
        affinity = CategoricalHyperparameter(
            name="affinity", choices=["euclidean", "manhattan", "cosine"], default_value="euclidean"
        )
        linkage = CategoricalHyperparameter(
            name="linkage", choices=["ward", "complete", "average"], default_value="ward"
        )
        pooling_func = CategoricalHyperparameter(
            "pooling_func", choices=["mean", "median", "max"], default_value="mean"
        )

        cs.add_hyperparameters([n_clusters, affinity, linkage, pooling_func])

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"),
        )
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs

    @staticmethod
    def get_pooling_function(method):
        """
        Maps the string pooling method to its corresponding numpy function.
        The hyperparameter is implemented with strings because it is easier to read in the sampling
        """
        pooling_func_mapping = dict(mean=np.mean, median=np.median, max=np.max)
        return pooling_func_mapping[method]

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        default_hyperparameters["n_clusters"] = min(default_hyperparameters["n_clusters"], self.x_train.shape[1])

        self.feature_preprocessor = FeatureAgglomeration(
            n_clusters=default_hyperparameters["n_clusters"],
            affinity=default_hyperparameters["affinity"],
            linkage=default_hyperparameters["linkage"],
            pooling_func=self.get_pooling_function(default_hyperparameters["pooling_func"])
        )

        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLKernelPCA(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/kernel_pca.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        random_state = Constant(name="random_state", value=SEED)

        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=10, upper=2000, default_value=100
        )
        kernel = CategoricalHyperparameter(
            name="kernel", choices=["poly", "rbf", "sigmoid", "cosine"], default_value="rbf"
        )
        gamma = UniformFloatHyperparameter(
            name="gamma",
            lower=3.0517578125e-05,
            upper=8,
            log=True,
            default_value=0.01,
        )
        degree = UniformIntegerHyperparameter(name="degree", lower=2, upper=5, default_value=3)
        coef0 = UniformFloatHyperparameter(name="coef0", lower=-1, upper=1, default_value=0)
        remove_zero_eig = Constant(name="remove_zero_eig", value="True")
        cs = ConfigurationSpace(seed=SEED)
        cs.add_hyperparameters([n_components, kernel, degree, gamma, coef0, random_state, remove_zero_eig])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        self.feature_preprocessor = KernelPCA(
            n_components=default_hyperparameters["n_components"],
            kernel=default_hyperparameters["kernel"],
            gamma=default_hyperparameters["gamma"],
            # degree=default_hyperparameters["degree"],
            # coef0=default_hyperparameters["coef0"],
            remove_zero_eig=default_hyperparameters["remove_zero_eig"],
            random_state=default_hyperparameters["random_state"]
        )
        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLRBFSampler(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/kitchen_sinks.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        random_state = Constant(name="random_state", value=SEED)
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=3.0517578125e-05, upper=8, default_value=1.0, log=True
        )
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=50, upper=10000, default_value=100, log=True
        )
        cs.add_hyperparameters([gamma, n_components, random_state])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        self.feature_preprocessor = RBFSampler(
            n_components=default_hyperparameters["n_components"],
            gamma=default_hyperparameters["gamma"],
            random_state=default_hyperparameters["random_state"]
        )
        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLNystroem(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/nystroem_sampler.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        random_state = Constant(name="random_state", value=SEED)
        kernel = CategoricalHyperparameter(name="kernel", choices=["poly", "rbf", "sigmoid", "cosine"],
                                           # chi2 is used only with sparse input
                                           default_value="rbf")
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=50, upper=10000, default_value=100, log=True
        )
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1
        )
        degree = UniformIntegerHyperparameter(name="degree", lower=2, upper=5, default_value=3)
        coef0 = UniformFloatHyperparameter(name="coef0", lower=-1, upper=1, default_value=0)

        cs.add_hyperparameters([random_state, kernel, degree, gamma, coef0, n_components])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])

        gamma_kernels = ["poly", "rbf", "sigmoid"]  # no chi2

        gamma_condition = InCondition(gamma, kernel, gamma_kernels)
        cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        self.feature_preprocessor = Nystroem(
            n_components=default_hyperparameters["n_components"],
            gamma=default_hyperparameters["gamma"],
            random_state=default_hyperparameters["random_state"],
            kernel=default_hyperparameters["kernel"],
            # degree=default_hyperparameters["degree"],
            # coef0=default_hyperparameters["coef0"],

        )

        self.x_train[self.x_train < 0] = 0.0
        self.x_test[self.x_test < 0] = 0.0

        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLPCA(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/pca.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        random_state = Constant(name="random_state", value=SEED)
        keep_variance = UniformFloatHyperparameter(
            # todo this is weird. its used to define the number of components. Check the provided link
            "keep_variance", 0.5, 0.9999, default_value=0.9999
        )
        whiten = CategoricalHyperparameter(
            "whiten", [False, True], default_value=False
        )
        copy = Constant(name="copy", value="True")
        cs.add_hyperparameters([random_state, keep_variance, whiten, copy])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        default_hyperparameters["keep_variance"] = float(default_hyperparameters["keep_variance"])

        self.feature_preprocessor = PCA(
            random_state=default_hyperparameters["random_state"],
            whiten=default_hyperparameters["whiten"],
            copy=default_hyperparameters["copy"],
            n_components=default_hyperparameters["keep_variance"]
        )

        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


#
class SKLPolynomialFeatures(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/polynomial.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter(
            "interaction_only", ["False", "True"], "False"
        )
        include_bias = CategoricalHyperparameter(
            "include_bias", ["True", "False"], "True"
        )
        cs.add_hyperparameters([degree, interaction_only, include_bias])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        self.feature_preprocessor = PolynomialFeatures(
            degree=default_hyperparameters["degree"],
            interaction_only=default_hyperparameters["interaction_only"],
            include_bias=default_hyperparameters["include_bias"]
        )

        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLRandomTreesEmbedding(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/random_trees_embedding.py
    """

    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        n_jobs = Constant(name="n_jobs", value=NJOBS)
        random_state = Constant(name="random_state", value=SEED)

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=10, upper=100, default_value=10
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=2, upper=10, default_value=5
        )
        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default_value=2
        )
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1
        )
        min_weight_fraction_leaf = Constant(name="min_weight_fraction_leaf", value=.0)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None"
        )
        sparse_output = Constant(name="sparse_output",
                                 value="False")  # todo how to push sparse output further in the pipeline?
        # bootstrap = CategoricalHyperparameter("bootstrap", [True, False]) # not used in Auto-Sklearn
        cs.add_hyperparameters(
            [
                n_estimators,
                max_depth,
                min_samples_split,
                min_samples_leaf,
                min_weight_fraction_leaf,
                max_leaf_nodes,
                # bootstrap,
                sparse_output,
                n_jobs,
                random_state
            ]
        )
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()

        self.feature_preprocessor = RandomTreesEmbedding(
            n_estimators=default_hyperparameters["n_estimators"],
            max_depth=default_hyperparameters["max_depth"],
            min_samples_split=default_hyperparameters["min_samples_split"],
            min_samples_leaf=default_hyperparameters["min_samples_leaf"],
            # min_weight_fraction_leaf=default_hyperparameters["min_weight_fraction_leaf"],
            max_leaf_nodes=default_hyperparameters["max_leaf_nodes"],
            sparse_output=default_hyperparameters["sparse_output"],
            n_jobs=default_hyperparameters["n_jobs"],
            random_state=default_hyperparameters["random_state"]
        )

        self.fit_transform_feature_preprocessor(x_and_y=False)
        self.sava_outputs()


class SKLSelectPercentile(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/select_percentile_classification.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        percentile = UniformFloatHyperparameter(
            name="percentile", lower=1, upper=99, default_value=50
        )
        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=["chi2", "f_classif", "mutual_info_classif"],
            default_value="chi2",
        )
        cs.add_hyperparameters([percentile, score_func])
        return cs

    @staticmethod
    def get_score_function(score_func):
        score_func_mapping = {
            "chi2": chi2,
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif
        }
        return score_func_mapping[score_func]

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()
        self.load_split_target_values()

        default_hyperparameters["score_func"] = self.get_score_function(default_hyperparameters["score_func"])

        self.feature_preprocessor = SelectPercentile(
            score_func=default_hyperparameters["score_func"],
            percentile=default_hyperparameters["percentile"]
        )
        self.x_train[self.x_train < 0] = 0.0
        self.fit_transform_feature_preprocessor(x_and_y=True)
        self.sava_outputs()


class SKLSelectRates(FeaturePreprocessor):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/feature_preprocessing/select_rates_classification.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.01, upper=0.5, default_value=0.1
        )

        score_func = CategoricalHyperparameter(
            name="score_func",
            choices=["chi2", "f_classif", "mutual_info_classif"],
            default_value="chi2",
        )
        mode = CategoricalHyperparameter(
            name="mode",
            choices=["fpr", "fdr", "fwe"],
            default_value="fpr",
        )
        cs.add_hyperparameters([alpha, score_func, mode])

        cond = NotEqualsCondition(mode, score_func, "mutual_info_classif")
        cs.add_condition(cond)
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_scaled_features()
        self.load_split_target_values()

        default_hyperparameters["score_func"] = self.get_score_function(default_hyperparameters["score_func"])

        self.feature_preprocessor = GenericUnivariateSelect(
            score_func=default_hyperparameters["score_func"],
            mode=default_hyperparameters["mode"],
            param=default_hyperparameters["alpha"]
        )
        self.x_train[self.x_train < 0] = 0.0
        self.fit_transform_feature_preprocessor(x_and_y=True)
        self.sava_outputs()

    @staticmethod
    def get_score_function(param):
        score_func_mapping = {
            "chi2": chi2,
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif
        }
        return score_func_mapping[param]


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
    y_predict = None
    accuracy = None

    def requires(self):
        return {
            "processed_features": self.processed_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "prediction": self.get_luigi_local_target_with_task_id("prediction.pkl"),
            "score": self.get_luigi_local_target_with_task_id("score.txt"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl")
        }

    def load_split_processed_features(self):
        self.x_train = pd.read_pickle(self.input()["processed_features"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["processed_features"]["x_test"].path)

    def load_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path)
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path)

    def sava_outputs(self):
        self.y_predict.to_pickle(self.output()["prediction"].path)

        with open(self.output()["score"].path, "w") as f:
            f.write(str(self.accuracy))

        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.estimator, outfile)

    def fit_predict_estimator(self):
        # self.convert_float64_to_float32(self.x_train)
        self.estimator.fit(self.x_train, self.y_train)

        # self.convert_float64_to_float32(self.x_test)
        self.y_predict = pd.DataFrame(
            columns=["y_predict"],
            data=self.estimator.predict(self.x_test))

    def compute_accuracy(self):
        self.accuracy = accuracy_score(self.y_test, self.y_predict)

    # def convert_float64_to_float32(self, df):
    #     column_data_types = df.dtypes
    #     for column in column_data_types.index:
    #         if column_data_types[column] == np.float64:
    #             return


class AdaBoost(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/adaboost.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False
        )
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True
        )
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R"
        )

        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False
        )
        random_state = Constant(name="random_state", value=SEED)

        base_estimator = Constant(name="base_estimator", value="DecisionTreeClassifier")

        cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth, random_state, base_estimator])
        return cs

    @staticmethod
    def get_base_estimator(estimator, kwargs):
        if estimator == "DecisionTreeClassifier":
            return DecisionTreeClassifier(**kwargs)
        else:
            raise ValueError("Unknown base_estimator %s" % estimator)

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        base_estimator = self.get_base_estimator(
            default_hyperparameters["base_estimator"],
            kwargs=dict(max_depth=default_hyperparameters["max_depth"]))

        self.estimator = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=default_hyperparameters["n_estimators"],
            learning_rate=default_hyperparameters["learning_rate"],
            algorithm=default_hyperparameters["algorithm"],
            random_state=default_hyperparameters["random_state"]
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLDecisionTree(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/adaboost.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)

        criterion = CategoricalHyperparameter(
            name="criterion", choices=["gini", "entropy"], default_value="gini")
        max_depth_factor = UniformFloatHyperparameter(
            name="max_depth_factor", lower=0.01, upper=2.0, default_value=0.5, log=True)

        min_samples_split = UniformIntegerHyperparameter(
            name="min_samples_split", lower=2, upper=20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1)
        min_weight_fraction_leaf = Constant(name="min_weight_fraction_leaf", value=.0)
        max_features = UnParametrizedHyperparameter(name="max_features", value=1.0)
        max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes", value="None")
        min_impurity_decrease = Constant(name="min_impurity_decrease", value=0.0)

        random_state = Constant(name="random_state", value=SEED)

        cs.add_hyperparameters(
            [criterion, max_depth_factor, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
             max_leaf_nodes, min_impurity_decrease, random_state])
        return cs

    @staticmethod
    def get_base_estimator(estimator, kwargs):
        if estimator == "DecisionTreeClassifier":
            return DecisionTreeClassifier(**kwargs)
        else:
            raise ValueError("Unknown base_estimator %s" % estimator)

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        num_features = self.x_train.shape[1]
        default_hyperparameters["max_depth_factor"] = int(default_hyperparameters["max_depth_factor"])

        default_hyperparameters["max_depth_factor"] = max(
            1, int(np.round(
                default_hyperparameters["max_depth_factor"] * num_features, 0))
        )

        self.estimator = DecisionTreeClassifier(
            criterion=default_hyperparameters["criterion"],
            max_depth=default_hyperparameters["max_depth_factor"],
            min_samples_split=default_hyperparameters["min_samples_split"],
            min_samples_leaf=default_hyperparameters["min_samples_leaf"],
            min_weight_fraction_leaf=default_hyperparameters["min_weight_fraction_leaf"],
            max_features=default_hyperparameters["max_features"],
            max_leaf_nodes=default_hyperparameters["max_leaf_nodes"],
            min_impurity_decrease=default_hyperparameters["min_impurity_decrease"],
            random_state=default_hyperparameters["random_state"],
            class_weight=None

        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


# class SKLExtraTrees(Classifier):
#     """
# https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/extra_trees.py    """
#     abstract = False
#
#     @staticmethod
#     def get_hyperparameter_search_space():
#         cs = ConfigurationSpace(seed=SEED)
#
#         criterion = CategoricalHyperparameter(
#             name="criterion", choices=["gini", "entropy"], default_value="gini")
#         max_features = UniformFloatHyperparameter(
#             name="max_features", lower=0.5, upper=1.0, default_value=0.5)
#         max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
#         min_samples_split = UniformIntegerHyperparameter(
#             name="min_samples_split", lower=2, upper=20, default_value=2)
#         min_samples_leaf = UniformIntegerHyperparameter(
#             name="min_samples_leaf", lower=1, upper=20, default_value=1)
#         min_weight_fraction_leaf = UnParametrizedHyperparameter(name="min_weight_fraction_leaf", value=.0)
#         max_leaf_nodes = UnParametrizedHyperparameter(name="max_leaf_nodes", value="None")
#         min_impurity_decrease = UnParametrizedHyperparameter(name="min_impurity_decrease", value=0.0)
#         random_state = Constant(name="random_state", value=SEED)
#         bootstrap = CategoricalHyperparameter(name="bootstrap", choices=["True", "False"], default_value="False")
#
#         cs.add_hyperparameters([criterion, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, random_state, bootstrap])
#         return cs
#
#     @staticmethod
#     def get_base_estimator(estimator, kwargs):
#         if estimator == "DecisionTreeClassifier":
#             return DecisionTreeClassifier(**kwargs)
#         else:
#             raise ValueError("Unknown base_estimator %s" % estimator)
#
#     def run(self):
#         default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
#         self.load_split_processed_features()
#         self.load_split_target_values()
#
#         # todo


class SKLGaussianNaiveBayes(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/gaussian_nb.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        var_smoothing = Constant(
            name="var_smoothing", value=1e-9)
        cs.add_hyperparameters([var_smoothing])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        self.estimator = GaussianNB(
            var_smoothing=default_hyperparameters["var_smoothing"]
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


# class SKLGradientBoosting(Classifier):
#     """
#     https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/gradient_boosting.py
#     """
#     abstract = False
#
#     @staticmethod
#     def get_hyperparameter_search_space():
#         cs = ConfigurationSpace(seed=SEED)
#         loss = Constant(name="loss", value="auto")
#         learning_rate = UniformFloatHyperparameter(
#             name="learning_rate", lower=0.01, upper=1.0, default_value=0.1,
#             log=True
#         )
#         min_samples_leaf = UniformIntegerHyperparameter(
#             name="min_samples_leaf", lower=1, upper=200, default_value=20,
#             log=True
#         )
#         max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
#         max_leaf_node = UniformIntegerHyperparameter(
#             name="max_leaf_node", lower=3, upper=2047, default_value=31, log=True
#         )
#         max_bins = Constant(name="max_bins", value=255)
#         l2_regularization = UniformIntegerHyperparameter(
#             name="l2_regularization", lower=1e-10, upper=1, default_value=1e-10, log=True
#         )
#         early_stop = CategoricalHyperparameter(
#             name="early_stop", choices=["off", "train", "valid", "both"], default_value="off"
#         )
#         tol = Constant(name="tol", value=1e-7)
#         scoring = Constant(name="scoring", value="loss")
#         n_iter_no_change = UniformIntegerHyperparameter(
#             name="n_iter_no_change", lower=1, upper=20, default_value=10
#         )
#         validation_fraction = UniformIntegerHyperparameter(
#             name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1
#         )
#
#         cs.add_hyperparameters(
#             [loss, learning_rate, min_samples_leaf, max_depth, max_leaf_node, max_bins, l2_regularization, early_stop,
#              tol, scoring, n_iter_no_change, validation_fraction])
#
#         n_iter_no_change_cond = InCondition(
#             n_iter_no_change, early_stop, ["valid", "train"]
#         )
#         validation_fraction_cond = EqualsCondition(
#             validation_fraction, early_stop, "valid"
#         )
#
#         cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])
#
#         return cs
#
#
#     def run(self):
#         # todo
#         default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
#         self.load_split_processed_features()
#         self.load_split_target_values()
#
#         self.estimator = GradientBoostingClassifier(
#
#         )
#
#         self.fit_predict_estimator()
#         self.compute_accuracy()
#         self.sava_outputs()


class SKLKNearestNeighbors(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/k_nearest_neighbors.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, default_value=1, log=True
        )
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform"
        )
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
        cs.add_hyperparameters([n_neighbors, weights, p])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        self.estimator = KNeighborsClassifier(
            n_neighbors=default_hyperparameters["n_neighbors"],
            weights=default_hyperparameters["weights"],
            p=default_hyperparameters["p"]
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()

class SKLLinearDiscriminantAnalysis(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/lda.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        shrinkage = CategoricalHyperparameter(
            name="shrinkage", choices=["auto", "manual"], default_value="auto"
        )
        shrinkage_factor = UniformFloatHyperparameter(
            name="shrinkage_factor", lower=0.0, upper=1.0, default_value=0.5
        )
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default_value=1e-4, log=True
        )
        cs.add_hyperparameters([shrinkage, shrinkage_factor, tol])
        cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        if default_hyperparameters["shrinkage"] is None:
            solver = "svd"
        elif (default_hyperparameters["shrinkage"] == "auto") or (default_hyperparameters["shrinkage"] == "manual"):
            solver = "lsqr"

        self.estimator = LinearDiscriminantAnalysis(
            shrinkage=default_hyperparameters["shrinkage"],
            tol=default_hyperparameters["tol"],
            solver=solver
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()

class SKLLinearSVC(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/liblinear_svc.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        penalty = CategoricalHyperparameter(
            name="penalty", choices=["l1", "l2"], default_value="l2"
        )
        loss = CategoricalHyperparameter(
            name="loss", choices=["hinge", "squared_hinge"], default_value="squared_hinge"
        )
        dual = Constant(name="dual", value="False")
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default_value=1e-4, log=True
        )
        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default_value=1.0
        )
        multi_class = Constant(name="multi_class", value="ovr")
        fit_intercept = Constant("fit_intercept", "True")
        intercept_scaling = Constant(name="intercept_scaling", value=1)
        random_state = Constant(name="random_state", value=SEED)

        cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class, intercept_scaling, random_state, fit_intercept])

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"), ForbiddenEqualsClause(loss, "hinge")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "hinge"),
        )
        penalty_and_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"), ForbiddenEqualsClause(penalty, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        cs.add_forbidden_clause(penalty_and_dual)

        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        self.estimator = LinearSVC(
            penalty=default_hyperparameters["penalty"],
            loss=default_hyperparameters["loss"],
            dual=default_hyperparameters["dual"],
            tol=default_hyperparameters["tol"],
            C=default_hyperparameters["C"],
            multi_class=default_hyperparameters["multi_class"],
            fit_intercept=default_hyperparameters["fit_intercept"],
            intercept_scaling=default_hyperparameters["intercept_scaling"],
            random_state=default_hyperparameters["random_state"],
            class_weight = None
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()

class SKLKernelSVC(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/libsvm_svc.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default_value=1.0
        )
        kernel = CategoricalHyperparameter(
            name="kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf"
        )
        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=3
        )
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter(
            name="coef0", lower=-1, upper=1, default_value=0
        )
        shrinking = CategoricalHyperparameter(
            name="shrinking", choices=["True", "False"], default_value="True"
        )
        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
        max_iter = UnParametrizedHyperparameter(name="max_iter", value=-1)
        random_state = Constant(name="random_state", value=SEED)

        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking, tol, max_iter, random_state])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()


        self.estimator = SVC(
            C=default_hyperparameters["C"],
            kernel=default_hyperparameters["kernel"],
            # degree=default_hyperparameters["degree"],
            gamma=default_hyperparameters["gamma"],
            # coef0=default_hyperparameters["coef0"],
            shrinking=default_hyperparameters["shrinking"],
            tol=default_hyperparameters["tol"],
            max_iter=default_hyperparameters["max_iter"],
            random_state=default_hyperparameters["random_state"],
            class_weight = None,
            decision_function_shape="ovr",
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


class SKLMultinominallNB(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/multinomial_nb.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-2, upper=100, default_value=1, log=True
        )
        fit_prior = CategoricalHyperparameter(
            name="fit_prior", choices=["True", "False"], default_value="True"
        )

        cs.add_hyperparameters([alpha, fit_prior])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        self.x_train[self.x_train < 0] = 0.0
        self.x_test[self.x_test < 0] = 0.0

        self.estimator = MultinomialNB(
            alpha=default_hyperparameters["alpha"],
            fit_prior=default_hyperparameters["fit_prior"]
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()

# class SKLPassiveAggressive(Classifier):
#     """
#     https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/passive_aggressive.py
#     """
#     abstract = False
#
#     @staticmethod
#     def get_hyperparameter_search_space():
#         cs = ConfigurationSpace(seed=SEED)
#         C = UniformFloatHyperparameter(
#             name="C", lower=0.03125, upper=32768, log=True, default_value=1.0
#         )
#         loss = CategoricalHyperparameter(
#             name="loss", choices=["hinge", "squared_hinge"], default_value="squared_hinge"
#         )
#         fit_intercept = Constant(name="fit_intercept", value="True")
#         tol = UniformFloatHyperparameter(
#             name="tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True
#         )
#         shuffle = Constant(name="shuffle", value="True")
#         max_iter = UnParametrizedHyperparameter(name="max_iter", value=1000)
#         random_state = Constant(name="random_state", value=SEED)
#
#         cs.add_hyperparameters([C, loss, fit_intercept, tol, shuffle, max_iter, random_state])
#
#         return cs
#
#     def run(self):
#         # todo


class SKLQuadraticDiscriminantAnalysis(Classifier):
    """
    https://github.com/automl/auto-sklearn/blob/development/autosklearn/pipeline/components/classification/qda.py
    """
    abstract = False

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace(seed=SEED)
        reg_param = UniformFloatHyperparameter(
            name="reg_param", lower=0.0, upper=1.0, default_value=0.0
        )
        cs.add_hyperparameters([reg_param])
        return cs

    def run(self):
        default_hyperparameters = self.get_default_hyperparameter_values_as_dict()
        self.load_split_processed_features()
        self.load_split_target_values()

        self.estimator = QuadraticDiscriminantAnalysis(
            reg_param=default_hyperparameters["reg_param"]
        )

        self.fit_predict_estimator()
        self.compute_accuracy()
        self.sava_outputs()


if __name__ == '__main__':

    from cls_luigi.repo_visualizer.static_json_repo import StaticJSONRepo
    from cls_luigi.repo_visualizer.dynamic_json_repo import DynamicJSONRepo

    target = Classifier.return_type()
    print("Collecting Repo")
    repository = RepoMeta.repository
    print("Build Repository...")
    # StaticJSONRepo(RepoMeta).dump_static_repo_json()

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Build Tree Grammar and inhabit Pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating results...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if actual > 0:
        max_results = actual

    validator = UniqueTaskPipelineValidator([Scaling, FeaturePreprocessor, Classifier])
    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    # results = [t() for t in
    #            inhabitation_result.evaluated[0:max_results]]  # this is what we should NOT be using in this case :)

    if results:
        DynamicJSONRepo(results).dump_dynamic_pipeline_json()
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        luigi.build(results, local_scheduler=True, detailed_summary=True)
    else:
        print("No results!")
