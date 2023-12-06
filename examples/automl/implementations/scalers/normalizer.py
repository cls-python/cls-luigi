from ..template import Scaler
from sklearn.preprocessing import Normalizer


class SKLNormalizer(Scaler):
    abstract = False

    def run(self):
        self.scaler = Normalizer(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()
