from ..template import FeaturePreprocessor
from sklearn.decomposition import KernelPCA

class SKLKernelPCA(FeaturePreprocessor):
    abstract = False

    def run(self):
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
