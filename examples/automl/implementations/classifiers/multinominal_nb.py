from sklearn.naive_bayes import MultinomialNB
from ..template import Classifier
import warnings
from examples.automl.utils.time_recorder import TimeRecorder
#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier

class SKLMultinomialNB(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_target_values()
                self._read_split_processed_features()

                self.x_train[self.x_train < 0] = 0.0
                self.x_test[self.x_test < 0] = 0.0

                self.estimator = MultinomialNB(
                    alpha=1,
                    fit_prior=True
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
