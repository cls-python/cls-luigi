from ..template import Scaler
from sklearn.preprocessing import StandardScaler


class SKLStandardScaler(Scaler):
    abstract = False

    def run(self):
        self.scaler = StandardScaler(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()
