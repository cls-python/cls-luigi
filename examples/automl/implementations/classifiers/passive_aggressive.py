from sklearn.linear_model import PassiveAggressiveClassifier
from ..template import Classifier
import warnings
from examples.automl.utils.time_recorder import TimeRecorder

#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier
class SKLPassiveAggressive(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_target_values()
                self._read_split_processed_features()

                self.estimator = PassiveAggressiveClassifier(
                    C=1.0,
                    fit_intercept=True,
                    loss="hinge",
                    tol=1e-4,
                    average=False,
                    shuffle=True,
                    random_state=self.global_params.seed,
                    warm_start=True
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
