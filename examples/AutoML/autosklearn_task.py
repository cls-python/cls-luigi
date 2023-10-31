from os import makedirs
from os.path import exists
from os.path import join as pjoin
import luigi
from cls_luigi import RESULTS_PATH
from cls_luigi.inhabitation_task import LuigiCombinator


class AutoSklearnTask(luigi.Task, LuigiCombinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not exists(RESULTS_PATH):
            makedirs(RESULTS_PATH)
        self.result_path = RESULTS_PATH

    @staticmethod
    def get_hyperparameter_search_space():
        return NotImplemented

    def get_default_hyperparameter_values_as_dict(self):
        default_hyperparameters = dict(self.get_hyperparameter_search_space().get_default_configuration())
        default_hyperparameters = self.handle_booleans_and_nones_in_config(default_hyperparameters)
        return default_hyperparameters

    def get_luigi_local_target_with_task_id(self, outfile, output_folder=RESULTS_PATH):
        return luigi.LocalTarget(pjoin(output_folder, self.task_id + "_" + outfile))

    @staticmethod
    def handle_booleans_and_nones_in_config(config):
        for key, value in config.items():
            if value == "True":
                config[key] = True
            elif value == "False":
                config[key] = False
            elif value == "None":
                config[key] = None
        return config
