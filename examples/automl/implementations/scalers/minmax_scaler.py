from ..template import Scaler
from sklearn.preprocessing import MinMaxScaler


class SKLMinMaxScaler(Scaler):
    abstract = False

    def run(self):
        self.scaler = MinMaxScaler(
            copy=False
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()
