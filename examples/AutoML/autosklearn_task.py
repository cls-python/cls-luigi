import logging
from logging import Logger
from os import makedirs
from os.path import exists
from os.path import join as pjoin
import luigi
from cls_luigi import RESULTS_PATH
from cls_luigi.inhabitation_task import LuigiCombinator


class AutoSklearnTask(luigi.Task, LuigiCombinator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not exists("logs"):
            makedirs("logs")


        # if not exists(RESULTS_PATH):
        #     makedirs(RESULTS_PATH)
        # self.result_path = RESULTS_PATH

    # @staticmethod
    # def get_hyperparameter_search_space():
    #     return NotImplemented
    #
    # def get_default_hyperparameter_values_as_dict(self):
    #     default_hyperparameters = dict(self.get_hyperparameter_search_space().get_default_configuration())
    #     default_hyperparameters = self.handle_booleans_and_nones_in_config(default_hyperparameters)
    #     return default_hyperparameters

    @staticmethod
    def makedirs_in_not_exist(path: str) -> None:
        if not exists(path):
            makedirs(path)

    def get_luigi_local_target_with_task_id(self, outfile: str, output_folder: str = RESULTS_PATH, dataset_name: str =None) -> luigi.LocalTarget:
        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)

        return luigi.LocalTarget(pjoin(dataset_outputs_folder, self.task_id + "_" + outfile))

    def get_luigi_local_target_without_task_id(self, outfile, output_folder=RESULTS_PATH, dataset_name=None):
        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)

        return luigi.LocalTarget(pjoin(dataset_outputs_folder, outfile))

    # @staticmethod
    # def handle_booleans_and_nones_in_config(config: dict):
    #     for key, value in config.items():
    #         if value == "True":
    #             config[key] = True
    #         elif value == "False":
    #             config[key] = False
    #         elif value == "None":
    #             config[key] = None
    #     return config

    def _log_warnings(self, warning_list: list) -> None:
        if len(warning_list) > 0:
            luigi_logger = self.get_luigi_logger()
            for w in warning_list:
                luigi_logger.warning("{}: {}".format(self.task_id, w.message))

    @staticmethod
    def get_luigi_logger() -> Logger:
        return logging.getLogger('luigi-root')
