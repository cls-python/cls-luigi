from ..template import FeaturePreprocessor
from sklearn.feature_selection import chi2, GenericUnivariateSelect
from examples.automl.utils.time_recorder import TimeRecorder
import warnings


class SKLSelectRates(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()
                self._read_split_target_values()

                self.feature_preprocessor = GenericUnivariateSelect(
                    score_func=chi2,
                    mode="fpr",
                    param=0.1
                )

                self.x_train[self.x_train < 0] = 0.0
                self.x_test[self.x_test < 0] = 0.0

                self.fit_transform_feature_preprocessor(x_and_y_required=True)
                self.sava_outputs()
