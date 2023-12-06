from ..template import FeaturePreprocessor
from sklearn.feature_selection import SelectPercentile, chi2


class SKLSelectPercentile(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()
        self._read_split_target_values()

        self.feature_preprocessor = SelectPercentile(
            score_func=chi2,
            percentile=50
        )
        self.x_train[self.x_train < 0] = 0.0
        self.x_test[self.x_test < 0] = 0.0

        self.fit_transform_feature_preprocessor(x_and_y_required=True)
        self.sava_outputs()
