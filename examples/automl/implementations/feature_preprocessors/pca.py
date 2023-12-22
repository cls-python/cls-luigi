from ..template import FeaturePreprocessor
from sklearn.decomposition import PCA
from examples.automl.utils.time_recorder import TimeRecorder
import warnings
class SKLPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()

                self.feature_preprocessor = PCA(
                    n_components=0.9999,
                    whiten=False,
                    copy=True,
                    random_state=self.global_params.seed,
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()
