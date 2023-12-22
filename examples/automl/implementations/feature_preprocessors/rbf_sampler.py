from ..template import FeaturePreprocessor
from sklearn.kernel_approximation import RBFSampler
from examples.automl.utils.time_recorder import TimeRecorder
import warnings

class SKLRBFSampler(FeaturePreprocessor):
    # aka kitchen sinks
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()

                self.feature_preprocessor = RBFSampler(
                    gamma=1.0,
                    n_components=100,
                    random_state=self.global_params.seed
                )
                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()
