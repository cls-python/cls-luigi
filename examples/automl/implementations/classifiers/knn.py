from sklearn.neighbors import KNeighborsClassifier
from ..template import Classifier

#TODO
# - this hanles only binary classification. We neeed to add OneVsRestClassifier
class SKLKNearestNeighbors(Classifier):
    abstract = False

    def run(self):
        self._read_split_processed_features()
        self._read_split_target_values()

        self.estimator = KNeighborsClassifier(
            n_neighbors=1,
            weights="uniform",
            p=2
        )

        self.fit_predict_estimator()
        self.create_run_summary()
        self.sava_outputs()
