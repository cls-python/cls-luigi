import warnings
from ..template import FeaturePreprocessor
from sklearn.decomposition import FastICA


class SKLFastICA(FeaturePreprocessor):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            self._read_split_scaled_features()

            self.feature_preprocessor = FastICA(
                algorithm="parallel",
                whiten=False,
                fun="logcosh",
                random_state=self.global_params.seed
            )

            self.fit_transform_feature_preprocessor(x_and_y_required=False)
            self.sava_outputs()
