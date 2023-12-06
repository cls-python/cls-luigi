from ..template import Classifier
from sklearn.naive_bayes import BernoulliNB

class SKLBernoulliNB(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = BernoulliNB(
            alpha=1.0,
            fit_prior=True,

        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
