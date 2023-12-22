from sklearn.svm import LinearSVC
from ..template import Classifier
import warnings
from examples.automl.utils.time_recorder import TimeRecorder
#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier
class SKLLinearSVC(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_target_values()

                self.estimator = LinearSVC(
                    penalty="l2",
                    loss="squared_hinge",
                    dual=False,
                    tol=1e-4,
                    C=1.0,
                    multi_class="ovr",
                    fit_intercept=True,
                    intercept_scaling=1,
                    random_state=self.global_params.seed
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
