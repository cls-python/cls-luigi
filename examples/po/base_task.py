import json
import pickle
from os import makedirs
from os.path import exists
from os.path import join as pjoin
import luigi

from cls_luigi import RESULTS_PATH
from cls_luigi.inhabitation_task import LuigiCombinator

from global_parameters import GlobalParameters


class BaseTaskClass(luigi.Task, LuigiCombinator):
    # worker_timeout = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_params = GlobalParameters()

    def get_luigi_local_target_with_task_id(self,
                                            outfile: str,
                                            output_folder: str = RESULTS_PATH,
                                            ) -> luigi.LocalTarget:

        if self.global_params.dataset_name != "None":
            output_folder = pjoin(output_folder, self.global_params.dataset_name)

        makedirs(output_folder, exist_ok=True)

        return luigi.LocalTarget(pjoin(output_folder, self.task_id + "_" + outfile))

    def get_luigi_local_target_without_task_id(self,
                                               outfile,
                                               output_folder=RESULTS_PATH,
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

    @staticmethod
    def dump_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def dump_json(obj, path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=5)
