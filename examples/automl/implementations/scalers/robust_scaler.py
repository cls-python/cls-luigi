from ..template import Scaler
from sklearn.preprocessing import RobustScaler


class SKLRobustScaler(Scaler):
    abstract = False

    def run(self):
        self.scaler = RobustScaler(
            copy=False,
            quantile_range=(0.25, 0.75)
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()
