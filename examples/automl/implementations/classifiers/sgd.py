from ..template import Classifier
from sklearn.linear_model import SGDClassifier
import warnings
from examples.automl.utils.time_recorder import TimeRecorder

class SKLSGD(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_target_values()
                self._read_split_processed_features()

                self.estimator = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0001,
                    l1_ratio=0.15,
                    fit_intercept=True,
                    tol=1e-4,
                    epsilon=1e-4,
                    learning_rate="invscaling",
                    eta0=0.01,
                    power_t=0.5,
                    average=False,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
