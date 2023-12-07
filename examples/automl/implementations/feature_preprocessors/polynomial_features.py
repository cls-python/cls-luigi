from ..template import FeaturePreprocessor
from sklearn.preprocessing import PolynomialFeatures


class SKLPolynomialFeatures(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = PolynomialFeatures(
            degree=2,
            interaction_only=False,
            include_bias=True
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()
