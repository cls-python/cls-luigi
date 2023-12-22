
from ..template import Classifier
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
from examples.automl.utils.time_recorder import TimeRecorder

class SKLGradientBoosting(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_target_values()
                self._read_split_processed_features()

                self.estimator = HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.1,
                    min_samples_leaf=20,
                    max_depth=None,
                    max_leaf_nodes=31,
                    max_bins=255,
                    l2_regularization=1e-10,
                    early_stopping=False,
                    tol=1e-7,
                    scoring="loss",
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                    random_state=self.global_params.seed,
                    warm_start=True
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
