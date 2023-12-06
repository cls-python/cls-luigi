from ..template import Scaler
from sklearn.preprocessing import QuantileTransformer

class SKLQuantileTransformer(Scaler):
    abstract = False

    def run(self):
        self.scaler = QuantileTransformer(
            copy=False,
            n_quantiles=1000,
            output_distribution="uniform",
            random_state=self.global_params.seed
        )
        self._read_split_imputed_features()
        self.fit_transform_scaler()
        self.sava_outputs()
