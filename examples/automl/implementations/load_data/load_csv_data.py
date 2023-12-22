import warnings
from sklearn.model_selection import train_test_split
from ..template import LoadAndSplitData
import pandas as pd

from examples.automl.utils.time_recorder import TimeRecorder


class LoadCSVData(LoadAndSplitData):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                x_train = pd.read_csv(self.global_params.x_train_path)
                x_test = pd.read_csv(self.global_params.x_test_path)
                y_train = pd.read_csv(self.global_params.y_train_path)
                y_test = pd.read_csv(self.global_params.y_test_path)

                x_train = x_train.reset_index(drop=True)
                x_test = x_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)

                x_train.to_pickle(self.output()["x_train"].path)
                x_test.to_pickle(self.output()["x_test"].path)
                y_train.to_pickle(self.output()["y_train"].path)
                y_test.to_pickle(self.output()["y_test"].path)

