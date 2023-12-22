from ..template import Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
from examples.automl.utils.time_recorder import TimeRecorder
#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier
class SKLLinearDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_processed_features()
                self._read_split_target_values()

                self.estimator = LinearDiscriminantAnalysis(
                    shrinkage=None,
                    solver="svd",
                    tol=1e-1,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
