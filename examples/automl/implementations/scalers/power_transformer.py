from ..template import Scaler
from sklearn.preprocessing import PowerTransformer

class SKLPowerTransformer(Scaler):
    abstract = False

    def run(self):
        self.scaler = PowerTransformer(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()
