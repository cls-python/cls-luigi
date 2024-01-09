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

    mode = luigi.TupleParameter(default=("cv"), significant=False)

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

    def run(self):
        if "cv" in self.mode:
            self._run_cv()

        if "fit" in self.mode:
            self._run_fit()

        if "predict" in self.mode:
            self._run_predict()

        if "infer" in self.mode:
            self._run_infer()

    def _run_cv(self):
        raise NotImplementedError

    def _run_fit(self):
        raise NotImplementedError

    def _run_predict(self):
        raise NotImplementedError

    def _run_infer(self):
        raise NotImplementedError

    def output(self):
        output_dict = {}

        if "cv" in self.mode:
            output_dict.update(self._cv_output())

        if "fit" in self.mode:
            output_dict.update(self._fit_output())

        if "predict" in self.mode:
            output_dict.update(self._predict_output())

        if "infer" in self.mode:
            output_dict.update(self._infer_output())

        return output_dict

    def _cv_output(self):
        raise NotImplementedError

    def _fit_output(self):
        raise NotImplementedError

    def _predict_output(self):
        raise NotImplementedError

    def _infer_output(self):
        raise NotImplementedError

    def _return_new_component_instance(self):
        return self._component()

    def _component(self):
        return NotImplementedError

    def _load_fitted_component(self):
        try:
            with open(self._fit_output()["fitted_component"].path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            print(f"Couldn't load component for task {self}.\n  Make sure pipeline is run with 'fit' mode first")
            raise e
