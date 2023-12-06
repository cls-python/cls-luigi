from ..template import FeaturePreprocessor
from sklearn.cluster import FeatureAgglomeration
import numpy as np

class SKLFeatureAgglomeration(FeaturePreprocessor):
    abstract = False

    def run(self):
        self._read_split_scaled_features()

        self.feature_preprocessor = FeatureAgglomeration(
            n_clusters=min(25, self.x_train.shape[1]),
            metric="euclidean",
            linkage="ward",
            pooling_func=np.mean
        )

        self.fit_transform_feature_preprocessor(x_and_y_required=False)
        self.sava_outputs()
