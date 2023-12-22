from logging import Logger, getLogger
from os import makedirs
from os.path import exists
from os.path import join as pjoin
import luigi
from cls_luigi import RESULTS_PATH
from cls_luigi.inhabitation_task import LuigiCombinator

from .global_parameters import GlobalParameters


class AutoSklearnTask(luigi.Task, LuigiCombinator):
    worker_timeout = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not exists("logs"):
            makedirs("logs")

        self.global_params = GlobalParameters()

    @staticmethod
    def makedirs_in_not_exist(path: str) -> None:
        if not exists(path):
            makedirs(path)

    def get_luigi_local_target_with_task_id(self,
                                            outfile: str,
                                            output_folder: str = RESULTS_PATH,
                                            dataset_name: int | str = None
                                            ) -> luigi.LocalTarget:

        if dataset_name is None:
            dataset_name = self.global_params.dataset_name

        dataset_name = self._check_if_int_and_cast_to_str(dataset_name)
        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)

        return luigi.LocalTarget(pjoin(dataset_outputs_folder, self.task_id + "_" + outfile))

    def get_luigi_local_target_without_task_id(self,
                                               outfile, output_folder=RESULTS_PATH,
                                               dataset_name: str | int = None
                                               ) -> luigi.LocalTarget:

        if dataset_name is None:
            dataset_name = self.global_params.dataset_name

        dataset_name = self._check_if_int_and_cast_to_str(dataset_name)

        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)

        return luigi.LocalTarget(pjoin(dataset_outputs_folder, outfile))

    @staticmethod
    def _check_if_int_and_cast_to_str(dataset_name: str | int) -> str:
        if isinstance(dataset_name, int):
            dataset_name = str(dataset_name)
        return dataset_name

    def _log_warnings(self, warning_list: list) -> None:
        if len(warning_list) > 0:
            luigi_logger = self.get_luigi_logger()
            for w in warning_list:
                luigi_logger.warning("{}: {}".format(self.task_id, w.message))

    @staticmethod
    def get_luigi_logger() -> Logger:
        return getLogger('luigi-root')

