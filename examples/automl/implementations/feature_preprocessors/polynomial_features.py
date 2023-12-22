from ..template import FeaturePreprocessor
from sklearn.preprocessing import PolynomialFeatures
from examples.automl.utils.time_recorder import TimeRecorder
import warnings


class SKLPolynomialFeatures(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_scaled_features()

                self.feature_preprocessor = PolynomialFeatures(
                    degree=2,
                    interaction_only=False,
                    include_bias=True
                )

                self.fit_transform_feature_preprocessor(x_and_y_required=False)
                self.sava_outputs()
