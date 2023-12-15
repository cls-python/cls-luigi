from ..template import Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier
class SKLLinearDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
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
