import logging
import sys
sys.path.append('..')
sys.path.append('../..')
# sys.path.append('/home/hadi/cls-luigi')


from luigi.execution_summary import execution_summary
execution_summary.summary_length = 10000


# CLS-Luigi imports
import subprocess

from cls_luigi.inhabitation_task import RepoMeta

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

# Global Parameters and AutoML validator
from implementations.global_parameters import GlobalParameters
from implementations.autosklearn_pipeline_validator import AutoMLPipelineValidator
from implementations.forbidden import FORBIDDEN

# template
from implementations.template import *

from time import time
import subprocess
from feature_type_analyzer import FeatureTypeAnalyzer
from download_and_save_openml_datasets import download_and_save_openml_dataset



def import_pipeline_components(
    include_categorical: bool = False,
    include_numerical: bool = True,
    include_string: bool = False) -> None: 
    
    # loading and splitting
    from implementations.load_and_split_data.load_and_split_pickled_tabular_data import LoadAndSplitPickledTabularData

    # category coalescence
    from implementations.category_coalescence.no_category_coalescence import NoCategoryCoalescence

    # encoding
    from implementations.encoding.no_encoding import NoEncoding


    # imputing
    from implementations.numerical_imputers.simple_imputer import SKLSimpleImpute

    # scaling
    from implementations.scalers.minmax_scaler import SKLMinMaxScaler
    from implementations.scalers.quantile_transformer import SKLQuantileTransformer
    from implementations.scalers.robust_scaler import SKLRobustScaler
    from implementations.scalers.standared_scaler import SKLStandardScaler
    from implementations.scalers.normalizer import SKLNormalizer
    from implementations.scalers.power_transformer import SKLPowerTransformer
    from implementations.scalers.no_scaling import NoScaling

    # feature preprocessing
    from implementations.feature_preprocessors.fast_ica import SKLFastICA
    from implementations.feature_preprocessors.feature_agglomeration import SKLFeatureAgglomeration
    from implementations.feature_preprocessors.kernel_pca import SKLKernelPCA
    from implementations.feature_preprocessors.no_feature_preprocessor import NoFeaturePreprocessor
    from implementations.feature_preprocessors.nystroem import SKLNystroem
    from implementations.feature_preprocessors.pca import SKLPCA
    from implementations.feature_preprocessors.polynomial_features import SKLPolynomialFeatures
    from implementations.feature_preprocessors.random_trees_embedding import SKLRandomTreesEmbedding
    from implementations.feature_preprocessors.rbf_sampler import SKLRBFSampler
    from implementations.feature_preprocessors.select_from_extra_trees_clf import SKLSelectFromExtraTrees
    from implementations.feature_preprocessors.select_from_svc_clf import SKLSelectFromLinearSVC
    from implementations.feature_preprocessors.select_percentile import SKLSelectPercentile
    from implementations.feature_preprocessors.select_rates import SKLSelectRates

    # classifier
    from implementations.classifiers.adaboost import SKLAdaBoost
    from implementations.classifiers.decision_tree import SKLDecisionTree
    from implementations.classifiers.random_forest import SKLRandomForest
    from implementations.classifiers.extra_trees import SKLExtraTrees
    from implementations.classifiers.gaussian_nb import SKLGaussianNaiveBayes
    from implementations.classifiers.gradient_boosting import SKLGradientBoosting
    from implementations.classifiers.knn import SKLKNearestNeighbors
    from implementations.classifiers.lda import SKLLinearDiscriminantAnalysis
    from implementations.classifiers.linear_svc import SKLLinearSVC
    from implementations.classifiers.multinominal_nb import SKLMultinomialNB
    from implementations.classifiers.passive_aggressive import SKLPassiveAggressive
    from implementations.classifiers.qda import SKLQuadraticDiscriminantAnalysis
    from implementations.classifiers.sgd import SKLSGD
    from implementations.classifiers.bernoulli_nb import SKLBernoulliNB
    from implementations.classifiers.svc import SKLKernelSVC
    
    
    if include_categorical is True:
        from implementations.encoding.ordinal_encoder import OrdinalEncoding
        from implementations.encoding.one_hot_encoder import OneHotEncoding
        from implementations.category_coalescence.minority_coalescence import MinorityCoalescence
        
        
    if include_string is True:
        pass




def main(ds_id: int, local_scheduler=True) -> None:
    X_path, y_path, ds_name= download_and_save_openml_dataset(ds_id)
    
    global_parameters = GlobalParameters()
    
    global_parameters.X_path = X_path
    global_parameters.y_path = y_path
    global_parameters.dataset_name = ds_name

    feature_type_analyzer = FeatureTypeAnalyzer(X_path) 
            
    import_pipeline_components(
        include_categorical=feature_type_analyzer.has_categorical_features()
        )
        

    target = Classifier.return_type()
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
        [LoadAndSplitData, CategoryCoalescer, CategoricalEncoder , NumericalImputer, Scaler, FeaturePreprocessor, Classifier])

    results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    automl_validator = AutoMLPipelineValidator(FORBIDDEN)

    results = [t for t in results if automl_validator.validate(t)]

    if results:
        print("Starting Luigid")
        loggers[1].warning("Starting Luigid")
        subprocess.run(["sudo", "luigid", "--background"])
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Running Pipelines")
        

        tick = time()

        luigi_run_result = luigi.build(results,
                                       local_scheduler=local_scheduler,
                                       detailed_summary=True,
                                       logging_conf_file="logging.conf",
                                       workers=2)

                                    #    logging_conf_file="/home/hadi/cls-luigi/examples/automl/logging.conf",
                                    #    workers =1)

        tock = time()

        with open("logs/{}_time.txt".format(ds_name), "w") as f:
            f.write("{} seconds".format(str(tock - tick)))

        print(luigi_run_result.summary_text)
        loggers[1].warning(luigi_run_result.summary_text)

        print("Done!")
        print("Killed Luigid")
        loggers[1].warning("Killed Luigid")
        subprocess.run(["sudo", "pkill", "-f", "luigid"])

    else:
        print("No results!")

    loggers[1].warning("\n{}\n{} This was dataset: {} {}\n{}\n".format(
        "*" * 150,
        "*" * 65,
        ds_name,
        "*" * (65 - len(str(ds_name))),
        "*" * 150))


if __name__ == "__main__":
    loggers = [logging.getLogger("luigi-root"), logging.getLogger("luigi-interface")]


    datasets = [
    # 361066,  # bank-marketing classification
    #146820,  # wilt classification
    #168868,  # APSFailure classification
    168911,  # jasmine classification
    # 168350,  # phoneme classification contains negative values
    # 359958,  # pc4 classification
    # 359962,  # kc1 classification
    # 359972,  # sylvin classification
    #359990,  # MiniBooNE classification
    # 146606,  #higgs
    ]
    


    for ds_id in datasets:
        main(ds_id, local_scheduler=False)

