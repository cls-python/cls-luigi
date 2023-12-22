import logging
import os
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
#from validators.no_duplicate_tasks_validator import NoDuplicateTasksValidator
from validators.not_forbidden_validator import NotForbiddenValidator

# template
from implementations.template import *

from time import time
import subprocess
from utils.feature_type_analyzer import FeatureTypeAnalyzer
from utils.download_and_save_openml_datasets import download_and_save_openml_dataset
from import_pipeline_components import import_pipeline_components

from utils.time_recorder import TimeRecorder


def main(
    ds_id: int,
    local_scheduler=True) -> None:

    x_train, x_test, y_train, y_test, ds_name = download_and_save_openml_dataset(ds_id)
    os.makedirs("logs", exist_ok=True)

    with TimeRecorder(f"logs/{ds_name}_time.json") as time_recorder:

        global_parameters = GlobalParameters()

        global_parameters.x_train_path = x_train
        global_parameters.x_test_path = x_test
        global_parameters.y_train_path = y_train
        global_parameters.y_test_path = y_test
        global_parameters.dataset_name = ds_name

        feature_type_analyzer = FeatureTypeAnalyzer(x_train)

        import_pipeline_components(
            include_categorical=feature_type_analyzer.has_categorical_features(),
            multiclass_classification=False
        )
        time_recorder.checkpoint("imported_components")

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
            [LoadAndSplitData, CategoryCoalescer, CategoricalEncoder, NumericalImputer, Scaler, FeaturePreprocessor,
             Classifier])

        results = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

        time_recorder.checkpoint("enumerated_results_with_UniqueTaskPipelineValidator")
        automl_validator = NotForbiddenValidator()

        results = [t for t in results if automl_validator.validate(t)]
        time_recorder.checkpoint("NotForbiddenValidator")

        if results:
            print("Starting Luigid")
            loggers[1].warning("Starting Luigid")
            subprocess.run(["luigid", "--background"])
            print("Number of results", max_results)
            print("Number of results after filtering", len(results))
            print("Running Pipelines")

            time_recorder.checkpoint("started_luigi_build")

            luigi_run_result = luigi.build(results,
                                           local_scheduler=local_scheduler,
                                           detailed_summary=True,
                                           logging_conf_file="logging.conf",
                                           workers=1)

            time_recorder.checkpoint("finished_luigi_build")

            print(luigi_run_result.summary_text)
            loggers[1].warning(luigi_run_result.summary_text)

            print("Done!")
            print("Killed Luigid")
            loggers[1].warning("Killed Luigid")
            subprocess.run(["pkill", "-f", "luigid"])

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
        359958,  # pc4 classification
        359962,  # kc1 classification
        361066,  # bank-marketing classification

        359972,  # sylvin classification
        146606,  #higgs
        168868,  # APSFailure classification



        # 146820,  # wilt classification
        # 168911,  # jasmine classification
        # 168350,  # phoneme classification contains negative values

        # 359990,  # MiniBooNE classification

    ]

    for ds_id in datasets:
        main(ds_id, local_scheduler=False)

