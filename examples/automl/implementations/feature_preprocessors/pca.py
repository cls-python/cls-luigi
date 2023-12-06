from ..template import FeaturePreprocessor
from sklearn.decomposition import PCA

class SKLPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = PCA(
            n_components=0.9999,
            whiten=False,
            copy=True,
            random_state=self.global_params.seed,
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()
