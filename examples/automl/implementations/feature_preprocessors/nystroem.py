from ..template import FeaturePreprocessor
from sklearn.kernel_approximation import Nystroem
from examples.automl.utils.time_recorder import TimeRecorder
import warnings


class SKLNystroem(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()

                self.feature_preprocessor = Nystroem(
                    kernel="rbf",
                    gamma=1.0,
                    coef0=0,
                    degree=3,
                    n_components=100,
                    random_state=self.global_params.seed

                )

                self.x_train[self.x_train < 0] = 0.0
                self.x_test[self.x_test < 0] = 0.0

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()
