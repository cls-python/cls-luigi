from ..template import Classifier
from sklearn.naive_bayes import GaussianNB

#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier
class SKLGaussianNaiveBayes(Classifier):
    abstract = False

    def run(self):
        self._read_split_target_values()
        self._read_split_processed_features()

        self.estimator = GaussianNB()

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
