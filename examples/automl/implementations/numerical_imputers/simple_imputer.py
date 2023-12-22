import warnings

from ..template import NumericalImputer
from sklearn.impute import SimpleImputer

from examples.automl.utils.time_recorder import TimeRecorder


class SKLSimpleImpute(NumericalImputer):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_features()

                self.imputer = SimpleImputer(
                    strategy="mean",
                    copy=False
                )

                self._fit_transform_imputer()
                self._save_outputs()
