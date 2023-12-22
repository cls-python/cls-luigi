import warnings

from ..template import FeaturePreprocessor
from sklearn.decomposition import KernelPCA

from examples.automl.utils.time_recorder import TimeRecorder


class SKLKernelPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()

                self.feature_preprocessor = KernelPCA(
                    n_components=100,
                    kernel="rbf",
                    gamma=0.1,
                    degree=3,
                    coef0=0,
                    remove_zero_eig=True,
                    random_state=self.global_params.seed,
                )
                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()
