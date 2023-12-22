from ..template import Scaler
from sklearn.preprocessing import QuantileTransformer
import warnings
from examples.automl.utils.time_recorder import TimeRecorder

class SKLQuantileTransformer(Scaler):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self.scaler = QuantileTransformer(
                    copy=False,
                    n_quantiles=1000,
                    output_distribution="uniform",
                    random_state=self.global_params.seed
                )
                self._read_split_imputed_features()
                self.fit_transform_scaler()
                self.sava_outputs()
